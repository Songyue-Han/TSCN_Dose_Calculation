from __future__ import annotations

import glob
import importlib.util
import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output"))
BUNDLED_MODEL_PATH = Path("/opt/app/bundled_model")


def resolve_model_path() -> Path:
    candidates = []
    env_model_path = os.environ.get("MODEL_PATH")
    if env_model_path:
        candidates.append(Path(env_model_path))
    candidates.extend([BUNDLED_MODEL_PATH, Path("/opt/ml/model")])

    for candidate in candidates:
        if (candidate / "5_train_LSTM_Res_Codex.py").exists() and (candidate / "best_model.pt").exists():
            return candidate
    return Path(env_model_path) if env_model_path else BUNDLED_MODEL_PATH


MODEL_PATH = resolve_model_path()
TASK = "photon-ct"
NUM_OUTPUT_FILES = 10
INPUT_DIR_BASE = "radiation-dose-calculation-source-ct-image"
INPUT_JSON_NAME = "stacked-photon-beam-level-metadata"
OUTPUT_DEPTH = 300
DEVICE = os.environ.get("DEVICE", "cuda:0")
RAW_RESAMPLE_CHUNK_Z = 32
BEV_PREPROCESS_Z_CHUNK_SIZE = int(os.environ.get("BEV_PREPROCESS_Z_CHUNK_SIZE", "64"))
INVERSE_SQUARE_MASTER_PATH = MODEL_PATH / "resources" / "inverse_square_master_normalized.npy"
WARMUP_INPUT_PATH = MODEL_PATH / "resources" / "warmup_input"
ENABLE_INIT_WARMUP = os.environ.get("ENABLE_INIT_WARMUP", "1") != "0"
ENABLE_DETAILED_TIMING = os.environ.get("ENABLE_DETAILED_TIMING", "0") != "0"
ENABLE_GPU_MINIMUM_CUTOFF = os.environ.get("ENABLE_GPU_MINIMUM_CUTOFF", "1") != "0"
ENABLE_CUDA_GRAPH = os.environ.get("ENABLE_CUDA_GRAPH", "0") != "0"
ENABLE_SHARED_SPLIT_HEAD = os.environ.get("ENABLE_SHARED_SPLIT_HEAD", "0") != "0"
ENABLE_SHARED_SEPARATE_HEAD_UPSAMPLE = os.environ.get("ENABLE_SHARED_SEPARATE_HEAD_UPSAMPLE", "1") != "0"
ENABLE_CHANNELS_LAST_2D = os.environ.get("ENABLE_CHANNELS_LAST_2D", "1") != "0"
ENABLE_CHANNELS_LAST_3D = os.environ.get("ENABLE_CHANNELS_LAST_3D", "0") != "0"
ENABLE_COMBINED_CT_DENSITY_SAMPLING = os.environ.get("ENABLE_COMBINED_CT_DENSITY_SAMPLING", "0") != "0"
ENABLE_PINNED_D2H = os.environ.get("ENABLE_PINNED_D2H", "1") != "0"
ENABLE_ASYNC_OUTPUT_WRITER = os.environ.get("ENABLE_ASYNC_OUTPUT_WRITER", "1") != "0"
ASYNC_OUTPUT_QUEUE_SIZE = int(os.environ.get("ASYNC_OUTPUT_QUEUE_SIZE", "3"))
ASYNC_OUTPUT_WRITER_THREADS = int(os.environ.get("ASYNC_OUTPUT_WRITER_THREADS", "1"))
ENABLE_TORCHSCRIPT_TRACE = os.environ.get("ENABLE_TORCHSCRIPT_TRACE", "0") != "0"
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "1") != "0"
TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")
TORCH_COMPILE_FULLGRAPH = os.environ.get("TORCH_COMPILE_FULLGRAPH", "1") != "0"
TORCH_COMPILE_DYNAMIC = os.environ.get("TORCH_COMPILE_DYNAMIC", "0") != "0"
STARTUP_WARMUP_CP_REPEATS = int(os.environ.get("STARTUP_WARMUP_CP_REPEATS", "2" if ENABLE_TORCH_COMPILE else "1"))
BODY_MASK_CT_THRESHOLD_HU = -1024.0


class TimingAccumulator:
    def __init__(self):
        self.values: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(seconds)

    def __getitem__(self, name: str) -> float:
        return self.values.get(name, 0.0)


def sync_if_cuda(device: torch.device, *, force: bool = False) -> None:
    if device.type == "cuda" and (force or ENABLE_DETAILED_TIMING):
        torch.cuda.synchronize(device)


def log_timing_line(prefix: str, values: dict[str, object]) -> None:
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    print(f"[timing] {prefix} " + " ".join(parts), flush=True)


def import_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TRAIN5 = import_module_from_path(MODEL_PATH / "5_train_LSTM_Res_Codex.py", "train5_res_submission")
PREP13 = import_module_from_path(MODEL_PATH / "1.3_New_Preprocessing.py", "bev_preprocessing_submission")
PRE1 = PREP13.PRE
PREP13.Z_CHUNK_SIZE = int(BEV_PREPROCESS_Z_CHUNK_SIZE)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def only_mha_in_dir(path: Path) -> Path:
    matches = glob.glob(str(path / "*.mha"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one .mha in {path}, found {len(matches)}: {matches}")
    return Path(matches[0])


def image_slot_dir(image_file_idx: int, input_path: Path = INPUT_PATH) -> Path:
    return input_path / "images" / f"{INPUT_DIR_BASE}-{image_file_idx + 1}"


def load_input_image(image_file_idx: int, input_path: Path = INPUT_PATH) -> sitk.Image:
    return sitk.ReadImage(str(only_mha_in_dir(image_slot_dir(image_file_idx, input_path))), sitk.sitkFloat32)


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0.0, atol=1e-6)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0.0, atol=1e-5)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0.0, atol=1e-6)
    )


def metaimage_number(value: float) -> str:
    return f"{float(value):.17g}"


def write_empty_stacked_mha(output_path: Path) -> None:
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 4",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
            "Offset = 0 0 0 0",
            "CenterOfRotation = 0 0 0 0",
            "ElementSpacing = 1 1 1 1",
            "DimSize = 1 1 1 1",
            "AnatomicalOrientation = ????",
            "ElementType = MET_FLOAT",
            "ElementDataFile = LOCAL",
            "",
        ]
    )
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        np.asarray([0.0], dtype=np.float32).tofile(f)


def stacked_mha_header_bytes(reference_img: sitk.Image, frame_count: int) -> bytes:
    size_xyz = reference_img.GetSize()
    spacing_xyz = reference_img.GetSpacing()
    origin_xyz = reference_img.GetOrigin()
    direction_3x3 = np.asarray(reference_img.GetDirection(), dtype=np.float64).reshape(3, 3)

    transform_4x4 = np.eye(4, dtype=np.float64)
    transform_4x4[:3, :3] = direction_3x3.T

    dim_size = (*size_xyz, int(frame_count))
    spacing = (*spacing_xyz, 1.0)
    offset = (*origin_xyz, 0.0)

    return "\n".join(
        [
            "ObjectType = Image",
            "NDims = 4",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = " + " ".join(metaimage_number(v) for v in transform_4x4.ravel()),
            "Offset = " + " ".join(metaimage_number(v) for v in offset),
            "CenterOfRotation = 0 0 0 0",
            "ElementSpacing = " + " ".join(metaimage_number(v) for v in spacing),
            "DimSize = " + " ".join(str(int(v)) for v in dim_size),
            "AnatomicalOrientation = ????",
            "ElementType = MET_FLOAT",
            "ElementDataFile = LOCAL",
            "",
        ]
    ).encode("ascii")


def _pwrite_all(fd: int, data: memoryview, offset: int) -> None:
    written = 0
    total = len(data)
    while written < total:
        n = os.pwrite(fd, data[written:], offset + written)
        if n <= 0:
            raise OSError(f"os.pwrite wrote {n} bytes")
        written += n


def output_frame_plan_from_metadata(metadata: list[dict]) -> dict[int, int]:
    slots: dict[int, set[int]] = {}
    for image_entry in metadata:
        for beam in image_entry.get("beams", []):
            for cp in beam.get("control_points", []):
                output_info = cp["output_info"]
                output_file_idx = int(output_info["output_file_idx"])
                idx_in_output = int(output_info["idx_in_output"])
                if output_file_idx < 0 or output_file_idx >= NUM_OUTPUT_FILES:
                    raise RuntimeError(f"output_file_idx out of range: {output_file_idx}")
                slots.setdefault(output_file_idx, set()).add(idx_in_output)

    frame_counts: dict[int, int] = {}
    for output_index, indices in slots.items():
        expected = list(range(max(indices) + 1))
        if sorted(indices) != expected:
            raise RuntimeError(
                f"Output slot {output_index} idx_in_output must be contiguous from 0; got {sorted(indices)}"
            )
        frame_counts[output_index] = len(expected)
    return frame_counts


class OutputSlotFile:
    def __init__(self, output_index: int, output_path: Path, reference_img: sitk.Image, frame_count: int):
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self.output_index = int(output_index)
        self.output_path = output_path
        self.reference_img = reference_img
        self.frame_count = int(frame_count)
        self.expected_shape_zyx = tuple(reversed(reference_img.GetSize()))
        self.one_volume_bytes = int(np.prod(self.expected_shape_zyx, dtype=np.int64)) * np.dtype(np.float32).itemsize
        self.header = stacked_mha_header_bytes(reference_img, self.frame_count)
        self.payload_offset = len(self.header)
        self.fd = os.open(str(output_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o666)
        try:
            os.write(self.fd, self.header)
            os.ftruncate(self.fd, self.payload_offset + self.frame_count * self.one_volume_bytes)
        except Exception:
            os.close(self.fd)
            raise

        self._lock = threading.Lock()
        self.submitted: set[int] = set()
        self.frames_written = 0
        self.bytes_written = 0
        self.worker_write_s = 0.0
        self.closed = False

    def validate_submit(self, idx: int, dose: np.ndarray, input_img: sitk.Image) -> None:
        if idx < 0 or idx >= self.frame_count:
            raise RuntimeError(f"Output slot {self.output_index} idx {idx} outside frame_count={self.frame_count}")
        if not same_geometry(input_img, self.reference_img):
            raise RuntimeError(f"Mixed image geometry in output slot {self.output_index}")
        if dose.shape != self.expected_shape_zyx:
            raise RuntimeError(
                f"Output slot {self.output_index} idx {idx} dose shape {dose.shape} "
                f"does not match CT zyx shape {self.expected_shape_zyx}"
            )
        self.submitted.add(idx)

    def write_frame(self, idx: int, dose: np.ndarray) -> float:
        t0 = time.perf_counter()
        arr = np.ascontiguousarray(dose, dtype=np.float32)
        view = memoryview(arr).cast("B")
        offset = self.payload_offset + int(idx) * self.one_volume_bytes
        _pwrite_all(self.fd, view, offset)
        write_s = time.perf_counter() - t0
        with self._lock:
            self.frames_written += 1
            self.bytes_written += self.one_volume_bytes
            self.worker_write_s += write_s
        return write_s

    def close(self) -> dict[str, object]:
        try:
            expected = set(range(self.frame_count))
            if self.submitted != expected:
                raise RuntimeError(
                    f"Output slot {self.output_index} submitted indices {sorted(self.submitted)}; expected {sorted(expected)}"
                )
            return self.summary()
        finally:
            self._close_fd()

    def _close_fd(self) -> None:
        if not self.closed:
            self.closed = True
            os.close(self.fd)

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "output_index": self.output_index,
                "frames": self.frames_written,
                "bytes_mb": self.bytes_written / 1.0e6,
                "worker_write_s": self.worker_write_s,
            }


class AsyncMHAOutputWriter:
    def __init__(self, metadata: list[dict], output_root: Path, queue_size: int):
        self.frame_counts = output_frame_plan_from_metadata(metadata)
        self.output_root = output_root
        self.queue_size = max(1, int(queue_size))
        self.writer_thread_count = max(1, int(ASYNC_OUTPUT_WRITER_THREADS))
        self.slot_files: dict[int, OutputSlotFile] = {}
        self.tasks: queue.Queue[tuple[OutputSlotFile, int, np.ndarray] | None] = queue.Queue(maxsize=self.queue_size)
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self.submit_block_s = 0.0
        self.worker_write_s = 0.0
        self.threads = [
            threading.Thread(target=self._worker, name=f"mha-writer-global-{idx}", daemon=False)
            for idx in range(self.writer_thread_count)
        ]
        for thread in self.threads:
            thread.start()

    def _output_path(self, output_index: int) -> Path:
        output_dir = self.output_root / "images" / f"stacked-radiation-dose-map-{output_index + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "output.mha"

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Async MHA writer failed") from self._error

    def submit(self, output_index: int, idx: int, dose: np.ndarray, input_img: sitk.Image) -> None:
        self.check_error()
        if output_index not in self.frame_counts:
            raise RuntimeError(f"Output slot {output_index} was not present in metadata output plan")
        slot_file = self.slot_files.get(output_index)
        if slot_file is None:
            slot_file = OutputSlotFile(
                output_index,
                self._output_path(output_index),
                input_img,
                self.frame_counts[output_index],
            )
            self.slot_files[output_index] = slot_file
        slot_file.validate_submit(idx, dose, input_img)
        t0 = time.perf_counter()
        self.tasks.put((slot_file, int(idx), dose))
        blocked_s = time.perf_counter() - t0
        with self._lock:
            self.submit_block_s += blocked_s
        self.check_error()

    def _worker(self) -> None:
        while True:
            item = self.tasks.get()
            try:
                if item is None:
                    return
                slot_file, idx, dose = item
                if self._error is not None:
                    continue
                write_s = slot_file.write_frame(idx, dose)
                with self._lock:
                    self.worker_write_s += write_s
            except BaseException as exc:
                self._error = exc
            finally:
                self.tasks.task_done()

    def close(self) -> dict[str, object]:
        for _ in self.threads:
            self.tasks.put(None)
        self.tasks.join()
        for thread in self.threads:
            thread.join()
        self.check_error()
        summaries = []
        for output_index, slot_file in sorted(self.slot_files.items()):
            summaries.append(slot_file.close())
        for output_index in range(NUM_OUTPUT_FILES):
            if output_index not in self.frame_counts:
                write_empty_stacked_mha(self._output_path(output_index))
        worker_write_s = sum(float(item["worker_write_s"]) for item in summaries)
        bytes_mb = sum(float(item["bytes_mb"]) for item in summaries)
        frames = sum(int(item["frames"]) for item in summaries)
        return {
            "async_output_submit_block_s": self.submit_block_s,
            "async_output_worker_write_s": worker_write_s,
            "async_output_bytes_mb": bytes_mb,
            "async_output_frames": frames,
            "slot_summaries": summaries,
        }

    def abort(self) -> None:
        if self._error is None:
            self._error = RuntimeError("Async MHA writer aborted")
        try:
            while True:
                self.tasks.get_nowait()
                self.tasks.task_done()
        except queue.Empty:
            pass
        for _ in self.threads:
            try:
                self.tasks.put_nowait(None)
            except queue.Full:
                pass
        for thread in self.threads:
            thread.join(timeout=2.0)
        for slot_file in self.slot_files.values():
            slot_file._close_fd()


def write_stacked_mha_streaming(
    output_path: Path,
    slot: dict[int, tuple[np.ndarray, sitk.Image, float]],
    output_index: int,
) -> None:
    expected = list(range(max(slot) + 1))
    if sorted(slot) != expected:
        raise RuntimeError(
            f"Output slot {output_index} idx_in_output must be contiguous from 0; got {sorted(slot)}"
        )

    reference_img = slot[0][1]
    size_xyz = reference_img.GetSize()
    header = stacked_mha_header_bytes(reference_img, len(expected))

    expected_shape_zyx = tuple(reversed(size_xyz))
    with open(output_path, "wb") as f:
        f.write(header)
        for idx in expected:
            dose, input_img, _ = slot[idx]
            if not same_geometry(input_img, reference_img):
                raise RuntimeError(f"Mixed image geometry in output slot {output_index}")
            if dose.shape != expected_shape_zyx:
                raise RuntimeError(
                    f"Output slot {output_index} idx {idx} dose shape {dose.shape} "
                    f"does not match CT zyx shape {expected_shape_zyx}"
                )
            np.ascontiguousarray(dose, dtype=np.float32).tofile(f)


def load_hu_density_curve() -> tuple[np.ndarray, np.ndarray]:
    params = load_json(MODEL_PATH / "resources" / "beam_parameters.json")
    entries = params["hu_to_density"]["entries"]
    hu = np.asarray([entry["hu"] for entry in entries], dtype=np.float32)
    density = np.asarray([entry["density_g_cm3"] for entry in entries], dtype=np.float32)
    order = np.argsort(hu)
    return hu[order], density[order]


def load_machine_geometry():
    params = load_json(MODEL_PATH / "resources" / "beam_parameters.json")["photon"]
    return PRE1.PhotonMachineGeometry(
        sad_mm=float(params["SAD_mm"]),
        jaw_x_mm=tuple(float(v) for v in params["jaw_x_mm"]),
        jaw_y_mm=tuple(float(v) for v in params["jaw_y_mm"]),
        mlc_num_leaf_pairs=int(params["mlc_num_leaf_pairs"]),
        mlc_leaf_thickness_mm=float(params["mlc_leaf_thickness_mm"]),
    )


def checkpoint_aperture_stats(checkpoint: dict) -> TRAIN5.ApertureNormalizationStats:
    return TRAIN5.ApertureNormalizationStats(
        log_area_mean=float(checkpoint["aperture_log_area_mean"]),
        log_area_std=float(checkpoint["aperture_log_area_std"]),
        log_s_mean=float(checkpoint["aperture_log_s_equivalent_mean"]),
        log_s_std=float(checkpoint["aperture_log_s_equivalent_std"]),
        min_input_value=float(checkpoint.get("aperture_log_min_value", TRAIN5.APERTURE_LOG_MIN_VALUE)),
        normalized_clamp=float(checkpoint.get("aperture_normalized_clamp", TRAIN5.APERTURE_NORMALIZED_CLAMP)),
    )


def set_model_inference_fast_mode(model: torch.nn.Module) -> None:
    module = model.module if hasattr(model, "module") else model
    if hasattr(module, "collect_diagnostics"):
        module.collect_diagnostics = False
    if hasattr(module, "use_channels_last_2d"):
        module.use_channels_last_2d = bool(ENABLE_CHANNELS_LAST_2D)
    if hasattr(module, "use_channels_last_3d"):
        module.use_channels_last_3d = bool(ENABLE_CHANNELS_LAST_3D)
    for submodule in module.modules():
        if hasattr(submodule, "collect_diagnostics"):
            submodule.collect_diagnostics = False
        if hasattr(submodule, "share_separate_head_upsample"):
            submodule.share_separate_head_upsample = bool(ENABLE_SHARED_SEPARATE_HEAD_UPSAMPLE)


def state_dict_for_requested_head_mode(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not ENABLE_SHARED_SPLIT_HEAD:
        return state_dict
    required_old = (
        "decoder.bev_head.conv200.weight",
        "decoder.bev_head.conv200.bias",
        "decoder.bev_head.out.weight",
        "decoder.bev_head.out.bias",
        "decoder.residual_head.conv200.weight",
        "decoder.residual_head.conv200.bias",
        "decoder.residual_head.out.weight",
        "decoder.residual_head.out.bias",
    )
    missing = [key for key in required_old if key not in state_dict]
    if missing:
        raise RuntimeError(f"Cannot map separate_standard checkpoint to shared_split head; missing={missing}")
    mapped = {key: value for key, value in state_dict.items()}
    mapped["decoder.output_heads.conv200.weight"] = torch.cat(
        (
            state_dict["decoder.bev_head.conv200.weight"],
            state_dict["decoder.residual_head.conv200.weight"],
        ),
        dim=0,
    ).contiguous()
    mapped["decoder.output_heads.conv200.bias"] = torch.cat(
        (
            state_dict["decoder.bev_head.conv200.bias"],
            state_dict["decoder.residual_head.conv200.bias"],
        ),
        dim=0,
    ).contiguous()
    mapped["decoder.output_heads.bev_out.weight"] = state_dict["decoder.bev_head.out.weight"].contiguous()
    mapped["decoder.output_heads.bev_out.bias"] = state_dict["decoder.bev_head.out.bias"].contiguous()
    mapped["decoder.output_heads.residual_out.weight"] = state_dict["decoder.residual_head.out.weight"].contiguous()
    mapped["decoder.output_heads.residual_out.bias"] = state_dict["decoder.residual_head.out.bias"].contiguous()
    for key in required_old:
        mapped.pop(key, None)
    return mapped


class ModelTraceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        ct: torch.Tensor,
        mlc: torch.Tensor,
        inverse_square: torch.Tensor,
        wed: torch.Tensor,
        aperture_features: torch.Tensor,
        angle_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            ct,
            mlc,
            inverse_square,
            wed,
            aperture_features=aperture_features,
            angle_features=angle_features,
        )


def build_patient_cache(image_file_idx: int, ct_img: sitk.Image, hu: np.ndarray, density_curve: np.ndarray, device: torch.device):
    ct_array = sitk.GetArrayFromImage(ct_img).astype(np.float32, copy=False)
    ct_gpu = torch.as_tensor(ct_array, dtype=torch.float32, device=device)
    density_gpu = PREP13.hu_to_mass_density_torch(ct_gpu, hu, density_curve)
    body_mask_gpu = TRAIN5.fill_binary_holes_2d_stack_gpu(ct_gpu > float(BODY_MASK_CT_THRESHOLD_HU)).contiguous()
    raw_shape_zyx = tuple(int(v) for v in ct_gpu.shape)
    raw_y = int(raw_shape_zyx[1])
    raw_x = int(raw_shape_zyx[2])
    raw_y_t = torch.arange(raw_y, dtype=torch.float32, device=device)
    raw_x_t = torch.arange(raw_x, dtype=torch.float32, device=device)
    raw_yy, raw_xx = torch.meshgrid(raw_y_t, raw_x_t, indexing="ij")

    non_air_zyx = torch.nonzero(ct_gpu > PREP13.BODY_THRESHOLD_HU, as_tuple=False)
    if non_air_zyx.numel() == 0:
        body_world_xyz = torch.empty((0, 3), dtype=torch.float32, device=device)
    else:
        origin, spacing, direction, _ = PREP13.image_geometry_torch(ct_img, device)
        index_xyz = torch.stack(
            (
                non_air_zyx[:, 2].to(dtype=torch.float32),
                non_air_zyx[:, 1].to(dtype=torch.float32),
                non_air_zyx[:, 0].to(dtype=torch.float32),
            ),
            dim=1,
        )
        body_world_xyz = (origin + (index_xyz * spacing) @ direction.T).contiguous()

    patient_cache = PREP13.PatientGpuCache(
        patient_id=f"image-{image_file_idx + 1}",
        ct_img=ct_img,
        mr_img=None,
        mr_ref_img=ct_img,
        ct_gpu=ct_gpu,
        mr_gpu=torch.empty((0,), dtype=torch.float32, device=device),
        density_gpu=density_gpu,
        ct_path=Path(f"input-image-{image_file_idx + 1}.mha"),
        mr_path=None,
        body_world_xyz=body_world_xyz,
        body_non_air_voxel_count=int(non_air_zyx.shape[0]),
    )
    patient_cache.body_mask_gpu = body_mask_gpu
    patient_cache.ct_density_delta_gpu = torch.stack(
        (
            ct_gpu - float(PREP13.CT_AIR_HU),
            density_gpu - float(density_curve[0]),
        ),
        dim=0,
    ).contiguous()
    patient_cache.ct_minus_air_gpu = patient_cache.ct_density_delta_gpu[0]
    patient_cache.density_minus_air_gpu = patient_cache.ct_density_delta_gpu[1]
    patient_cache.raw_shape_zyx = raw_shape_zyx
    patient_cache.raw_yy = raw_yy.contiguous()
    patient_cache.raw_xx = raw_xx.contiguous()
    return patient_cache


def sample_precomputed_delta_with_grid(
    volume_delta_1cdhw: torch.Tensor,
    sampling_grid_1dhw3: torch.Tensor,
    cval: float,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if chunk_size is None:
        chunk_size = int(BEV_PREPROCESS_Z_CHUNK_SIZE)
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for z0 in range(0, sampling_grid_1dhw3.shape[1], chunk_size):
            grid = sampling_grid_1dhw3[:, z0:z0 + chunk_size]
            sampled = F.grid_sample(
                volume_delta_1cdhw,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0] + float(cval)
            outputs.append(sampled)
    return torch.cat(outputs, dim=0)


def sample_precomputed_delta_channels_with_grid(
    volume_delta_1cdhw: torch.Tensor,
    sampling_grid_1dhw3: torch.Tensor,
    cvals: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if chunk_size is None:
        chunk_size = int(BEV_PREPROCESS_Z_CHUNK_SIZE)
    outputs: list[torch.Tensor] = []
    cvals_view = cvals.to(device=volume_delta_1cdhw.device, dtype=volume_delta_1cdhw.dtype).reshape(-1, 1, 1, 1)
    with torch.inference_mode():
        for z0 in range(0, sampling_grid_1dhw3.shape[1], chunk_size):
            grid = sampling_grid_1dhw3[:, z0:z0 + chunk_size]
            sampled = F.grid_sample(
                volume_delta_1cdhw,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0] + cvals_view
            outputs.append(sampled)
    return torch.cat(outputs, dim=1)


def prediction_to_raw_grid_cached(
    parameter_map: torch.Tensor,
    patient_cache,
    ct_img: sitk.Image,
    geometry,
    output_start: int,
    precomp: dict[str, np.ndarray],
) -> torch.Tensor:
    metadata = {
        "geometry": asdict(geometry),
        "output_start_index": int(output_start),
    }
    affine_np = TRAIN5.raw_index_to_bev_affine(ct_img, metadata, precomp)
    affine = torch.as_tensor(affine_np, dtype=torch.float32, device=parameter_map.device)
    raw_shape_zyx = getattr(patient_cache, "raw_shape_zyx", tuple(reversed(ct_img.GetSize())))
    raw_z, raw_y, raw_x = (int(v) for v in raw_shape_zyx)
    yy = getattr(patient_cache, "raw_yy", None)
    xx = getattr(patient_cache, "raw_xx", None)
    if yy is None or xx is None:
        y = torch.arange(raw_y, dtype=parameter_map.dtype, device=parameter_map.device)
        x = torch.arange(raw_x, dtype=parameter_map.dtype, device=parameter_map.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
    chunks: list[torch.Tensor] = []
    for z0 in range(0, raw_z, RAW_RESAMPLE_CHUNK_Z):
        z1 = min(z0 + RAW_RESAMPLE_CHUNK_Z, raw_z)
        sampled, _ = TRAIN5.sample_trilinear_parameter_map_chunk(
            parameter_map,
            z0,
            z1,
            raw_y,
            raw_x,
            affine,
            yy,
            xx,
        )
        chunks.append(sampled)
    return torch.cat(chunks, dim=0)


class DoseRadResAlgorithm:
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        checkpoint_path = MODEL_PATH / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.dose_scale = float(checkpoint.get("dose_scale", TRAIN5.DOSE_SCALE))
        self.aperture_stats = checkpoint_aperture_stats(checkpoint)
        if ENABLE_SHARED_SPLIT_HEAD:
            TRAIN5.OUTPUT_HEAD_MODE = "shared_split"
        self.model = TRAIN5.build_model().to(self.device).eval()
        self.model.load_state_dict(
            state_dict_for_requested_head_mode(checkpoint["model_state_dict"]),
            strict=True,
        )
        set_model_inference_fast_mode(self.model)
        self.eager_model = self.model
        self.model_is_torchscript = False
        self.model_is_torch_compiled = False
        if ENABLE_TORCHSCRIPT_TRACE:
            trace_t0 = time.perf_counter()
            try:
                example_inputs = (
                    torch.zeros((1, OUTPUT_DEPTH, 200, 200), dtype=torch.float32, device=self.device),
                    torch.zeros((1, OUTPUT_DEPTH, 200, 200), dtype=torch.float32, device=self.device),
                    torch.zeros((1, OUTPUT_DEPTH, 200, 200), dtype=torch.float32, device=self.device),
                    torch.zeros((1, OUTPUT_DEPTH, 200, 200), dtype=torch.float32, device=self.device),
                    torch.zeros((1, 2), dtype=torch.float32, device=self.device),
                    torch.zeros((1, 2), dtype=torch.float32, device=self.device),
                )
                traced = torch.jit.trace(ModelTraceWrapper(self.model), example_inputs, strict=False)
                self.model = traced.eval()
                self.model_is_torchscript = True
                sync_if_cuda(self.device, force=True)
                log_timing_line("torchscript_trace", {"status": "ok", "trace_s": time.perf_counter() - trace_t0})
            except Exception as exc:
                sync_if_cuda(self.device, force=True)
                log_timing_line("torchscript_trace", {"status": "disabled_fallback", "trace_s": time.perf_counter() - trace_t0, "error": repr(exc)})
        if ENABLE_TORCH_COMPILE and not self.model_is_torchscript:
            compile_t0 = time.perf_counter()
            try:
                self.model = torch.compile(
                    self.model,
                    mode=TORCH_COMPILE_MODE,
                    fullgraph=bool(TORCH_COMPILE_FULLGRAPH),
                    dynamic=bool(TORCH_COMPILE_DYNAMIC),
                )
                self.model_is_torch_compiled = True
                log_timing_line(
                    "torch_compile",
                    {
                        "status": "enabled",
                        "setup_s": time.perf_counter() - compile_t0,
                        "mode": TORCH_COMPILE_MODE,
                        "fullgraph": int(TORCH_COMPILE_FULLGRAPH),
                        "dynamic": int(TORCH_COMPILE_DYNAMIC),
                    },
                )
            except Exception as exc:
                self.model_is_torch_compiled = False
                log_timing_line(
                    "torch_compile",
                    {
                        "status": "disabled_fallback",
                        "setup_s": time.perf_counter() - compile_t0,
                        "error": repr(exc),
                    },
                )
        self.cuda_graph_enabled = bool(ENABLE_CUDA_GRAPH and self.device.type == "cuda")
        self.cuda_graph_failed = False
        self.cuda_graph = None
        self.cuda_graph_inputs = None
        self.cuda_graph_aperture_features = None
        self.cuda_graph_angle_features = None
        self.cuda_graph_outputs = None
        checkpoint_epoch = checkpoint.get("epoch")
        checkpoint_best_val_loss = checkpoint.get("best_val_loss")
        del checkpoint

        self.precomp = PREP13.load_precompute()
        self.machine = load_machine_geometry()
        self.hu, self.density_curve = load_hu_density_curve()
        self.rho_air = float(self.density_curve[0])
        self.wed_geometry = PREP13.build_wed_geometry_gpu(self.precomp, self.device)
        self.mlc_overlap_y_master = PREP13.load_mlc_overlap_y_master_gpu(self.precomp, self.device)
        self.inverse_square_master_t = self.load_inverse_square_master()

        self.z_master = self.precomp["master_z_centers_mm"].astype(np.float32)
        self.y_centers = self.precomp["cart_y_centers_mm"].astype(np.float32)
        self.x_centers = self.precomp["cart_x_centers_mm"].astype(np.float32)

        print(
            f"Loaded checkpoint {checkpoint_path} "
            f"epoch={checkpoint_epoch} best_val_loss={checkpoint_best_val_loss}"
        )
        log_timing_line(
            "runtime_options",
            {
                "channels_last_2d": int(ENABLE_CHANNELS_LAST_2D),
                "channels_last_3d": int(ENABLE_CHANNELS_LAST_3D),
                "shared_split_head": int(ENABLE_SHARED_SPLIT_HEAD),
                "shared_separate_head_upsample": int(ENABLE_SHARED_SEPARATE_HEAD_UPSAMPLE),
                "cuda_graph": int(ENABLE_CUDA_GRAPH),
                "bev_preprocess_z_chunk": int(BEV_PREPROCESS_Z_CHUNK_SIZE),
                "gpu_minimum_cutoff": int(ENABLE_GPU_MINIMUM_CUTOFF),
                "combined_ct_density_sampling": int(ENABLE_COMBINED_CT_DENSITY_SAMPLING),
                "pinned_d2h": int(ENABLE_PINNED_D2H),
                "async_output_writer": int(ENABLE_ASYNC_OUTPUT_WRITER),
                "async_output_queue_size": int(ASYNC_OUTPUT_QUEUE_SIZE),
                "async_output_writer_threads": int(ASYNC_OUTPUT_WRITER_THREADS),
                "torchscript_trace": int(self.model_is_torchscript),
                "torch_compile": int(self.model_is_torch_compiled),
                "torch_compile_mode": TORCH_COMPILE_MODE,
                "torch_compile_fullgraph": int(TORCH_COMPILE_FULLGRAPH),
                "torch_compile_dynamic": int(TORCH_COMPILE_DYNAMIC),
                "startup_warmup_cp_repeats": int(STARTUP_WARMUP_CP_REPEATS),
            },
        )
        if ENABLE_INIT_WARMUP:
            self.run_startup_warmup()

    def model_forward(self, model_inputs, aperture_features: torch.Tensor, angle_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_is_torchscript:
            return self.model(*model_inputs, aperture_features, angle_features)
        if self.model_is_torch_compiled:
            try:
                return self.model(*model_inputs, aperture_features=aperture_features, angle_features=angle_features)
            except Exception as exc:
                self.model_is_torch_compiled = False
                self.model = self.eager_model
                print(f"[timing] torch_compile status=disabled_fallback_at_forward error={repr(exc)}", flush=True)
                return self.model(*model_inputs, aperture_features=aperture_features, angle_features=angle_features)
        if not self.cuda_graph_enabled or self.cuda_graph_failed:
            return self.model(*model_inputs, aperture_features=aperture_features, angle_features=angle_features)

        try:
            if self.cuda_graph is None:
                self.cuda_graph_inputs = tuple(torch.empty_like(tensor) for tensor in model_inputs)
                self.cuda_graph_aperture_features = torch.empty_like(aperture_features)
                self.cuda_graph_angle_features = torch.empty_like(angle_features)
                for dst, src in zip(self.cuda_graph_inputs, model_inputs):
                    dst.copy_(src)
                self.cuda_graph_aperture_features.copy_(aperture_features)
                self.cuda_graph_angle_features.copy_(angle_features)

                # Warm up with static buffers before graph capture so cuDNN chooses algorithms outside capture.
                for _ in range(2):
                    self.model(
                        *self.cuda_graph_inputs,
                        aperture_features=self.cuda_graph_aperture_features,
                        angle_features=self.cuda_graph_angle_features,
                    )
                sync_if_cuda(self.device, force=True)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.cuda_graph_outputs = self.model(
                        *self.cuda_graph_inputs,
                        aperture_features=self.cuda_graph_aperture_features,
                        angle_features=self.cuda_graph_angle_features,
                    )
                self.cuda_graph = graph
            else:
                if self.cuda_graph_inputs is None or self.cuda_graph_aperture_features is None or self.cuda_graph_angle_features is None:
                    raise RuntimeError("CUDA graph buffers are not initialized")
                for dst, src in zip(self.cuda_graph_inputs, model_inputs):
                    dst.copy_(src)
                self.cuda_graph_aperture_features.copy_(aperture_features)
                self.cuda_graph_angle_features.copy_(angle_features)
                self.cuda_graph.replay()
            if self.cuda_graph_outputs is None:
                raise RuntimeError("CUDA graph outputs are not initialized")
            return self.cuda_graph_outputs
        except Exception as exc:
            self.cuda_graph_failed = True
            self.cuda_graph_enabled = False
            print(f"[timing] cuda_graph status=disabled_fallback error={repr(exc)}", flush=True)
            return self.model(*model_inputs, aperture_features=aperture_features, angle_features=angle_features)

    def load_inverse_square_master(self) -> torch.Tensor:
        load_t0 = time.perf_counter()
        if INVERSE_SQUARE_MASTER_PATH.exists():
            inverse_square_master = np.load(INVERSE_SQUARE_MASTER_PATH).astype(np.float32, copy=False)
        else:
            inverse_square_master = TRAIN5.normalize_inverse_square(
                PREP13.build_inverse_square_master_gpu(self.precomp, torch.device("cpu")).numpy()
            ).astype(np.float32, copy=False)
        tensor = torch.as_tensor(inverse_square_master, dtype=torch.float32, device=self.device)
        sync_if_cuda(self.device, force=True)
        log_timing_line(
            "inverse_square_master_load",
            {
                "source": INVERSE_SQUARE_MASTER_PATH.name if INVERSE_SQUARE_MASTER_PATH.exists() else "generated",
                "load_to_gpu_s": time.perf_counter() - load_t0,
                "shape": "x".join(str(v) for v in tensor.shape),
            },
        )
        return tensor

    def run_startup_warmup(self) -> None:
        warmup_t0 = time.perf_counter()
        try:
            metadata_path = WARMUP_INPUT_PATH / f"{INPUT_JSON_NAME}.json"
            metadata = load_json(metadata_path)
            image_entry = metadata[0]
            image_idx = int(image_entry["image_file_idx"])
            ct_img = load_input_image(image_idx, WARMUP_INPUT_PATH)
            patient_cache = build_patient_cache(
                image_idx,
                ct_img,
                self.hu,
                self.density_curve,
                self.device,
            )
            beam = image_entry["beams"][0]
            if "iso_center" not in beam and "iso_center" in image_entry:
                beam = {**beam, "iso_center": image_entry["iso_center"]}
            cp = beam["control_points"][0]
            for _ in range(max(1, int(STARTUP_WARMUP_CP_REPEATS))):
                self.predict_control_point(patient_cache, ct_img, beam, cp)
            sync_if_cuda(self.device, force=True)
            log_timing_line(
                "startup_warmup",
                {
                    "status": "ok",
                    "warmup_s": time.perf_counter() - warmup_t0,
                    "cp_repeats": int(STARTUP_WARMUP_CP_REPEATS),
                },
            )
        except Exception as exc:
            log_timing_line(
                "startup_warmup",
                {"status": "failed", "warmup_s": time.perf_counter() - warmup_t0, "error": repr(exc)},
            )
            raise

    def predict_control_point(
        self,
        patient_cache,
        ct_img: sitk.Image,
        beam: dict,
        cp: dict,
        minimum_cutoff: float | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        timing = TimingAccumulator()
        step_t0 = time.perf_counter()
        geometry = PRE1.geometry_from_beam_cp(beam, cp)
        timing.add("geometry_s", time.perf_counter() - step_t0)

        sync_if_cuda(self.device)
        step_t0 = time.perf_counter()
        offset_info = PREP13.compute_output_window_from_patient_cache(
            patient_cache,
            geometry,
            self.z_master,
            self.device,
        )
        sync_if_cuda(self.device)
        timing.add("z_offset_s", time.perf_counter() - step_t0)
        output_start = int(offset_info["output_start_index"])
        output_end = int(offset_info["output_end_index"])
        z_output = self.z_master[output_start:output_end + 1]

        sync_if_cuda(self.device)
        step_t0 = time.perf_counter()
        ct_master_grid = PREP13.build_bev_sampling_grid_gpu(
            ct_img,
            geometry,
            self.z_master[:output_end + 1],
            self.y_centers,
            self.x_centers,
            device=self.device,
            chunk_size=int(BEV_PREPROCESS_Z_CHUNK_SIZE),
        )
        sync_if_cuda(self.device)
        timing.add("bev_grid_build_s", time.perf_counter() - step_t0)
        output_grid_ct = ct_master_grid[:, output_start:output_end + 1]

        if ENABLE_COMBINED_CT_DENSITY_SAMPLING:
            sync_if_cuda(self.device)
            step_t0 = time.perf_counter()
            combined_master = sample_precomputed_delta_channels_with_grid(
                patient_cache.ct_density_delta_gpu[None],
                ct_master_grid,
                cvals=torch.tensor((float(PREP13.CT_AIR_HU), float(self.rho_air)), dtype=torch.float32, device=self.device),
            )
            ct_bev = combined_master[0, output_start:output_end + 1].clamp_(PREP13.CT_AIR_HU, 3071.0)
            density_master_needed = combined_master[1]
            sync_if_cuda(self.device)
            timing.add("ct_density_combined_s", time.perf_counter() - step_t0)
            timing.add("ct_bev_s", 0.0)
            timing.add("density_master_s", 0.0)
        else:
            sync_if_cuda(self.device)
            step_t0 = time.perf_counter()
            ct_bev = sample_precomputed_delta_with_grid(
                patient_cache.ct_minus_air_gpu[None, None],
                output_grid_ct,
                cval=PREP13.CT_AIR_HU,
            ).clamp_(PREP13.CT_AIR_HU, 3071.0)
            sync_if_cuda(self.device)
            timing.add("ct_bev_s", time.perf_counter() - step_t0)

            sync_if_cuda(self.device)
            step_t0 = time.perf_counter()
            density_master_needed = sample_precomputed_delta_with_grid(
                patient_cache.density_minus_air_gpu[None, None],
                ct_master_grid,
                cval=self.rho_air,
            )
            sync_if_cuda(self.device)
            timing.add("density_master_s", time.perf_counter() - step_t0)

        sync_if_cuda(self.device)
        step_t0 = time.perf_counter()
        wed_bev = PREP13.compute_wed(
            density_master_needed,
            output_start,
            output_end,
            self.wed_geometry,
            self.rho_air,
            self.device,
        )
        sync_if_cuda(self.device)
        timing.add("wed_s", time.perf_counter() - step_t0)

        mlc_overlap_y = None
        if self.mlc_overlap_y_master is not None:
            mlc_overlap_y = self.mlc_overlap_y_master[output_start:output_end + 1]
        sync_if_cuda(self.device)
        step_t0 = time.perf_counter()
        mlc_fraction = PREP13.exact_mlc_area_fraction_shifted(
            cp,
            self.machine,
            z_output,
            self.y_centers,
            self.x_centers,
            self.device,
            overlap_y_y_leaf=mlc_overlap_y,
        )
        sync_if_cuda(self.device)
        timing.add("mlc_s", time.perf_counter() - step_t0)

        step_t0 = time.perf_counter()
        a_open, _, s_equiv = PREP13.mlc_aperture_scalars(cp, self.machine)
        aperture_features = torch.as_tensor(
            TRAIN5.normalize_aperture_features(a_open, s_equiv, self.aperture_stats)[None],
            dtype=torch.float32,
            device=self.device,
        )
        angle_features = torch.as_tensor(
            TRAIN5.gantry_angle_features(float(cp["gantry_angle"]))[None],
            dtype=torch.float32,
            device=self.device,
        )
        sync_if_cuda(self.device)
        timing.add("conditioning_s", time.perf_counter() - step_t0)

        step_t0 = time.perf_counter()
        inverse_square = self.inverse_square_master_t[output_start:output_end + 1]
        sync_if_cuda(self.device)
        timing.add("inverse_square_s", time.perf_counter() - step_t0)

        step_t0 = time.perf_counter()
        model_inputs = (
            (ct_bev * float(TRAIN5.CT_SCALE))[None],
            (mlc_fraction * float(TRAIN5.PROJ_SCALE))[None],
            inverse_square[None],
            (wed_bev / float(TRAIN5.WED_SCALE_MM_WATER))[None],
        )
        sync_if_cuda(self.device)
        timing.add("input_pack_s", time.perf_counter() - step_t0)

        with torch.inference_mode():
            sync_if_cuda(self.device)
            step_t0 = time.perf_counter()
            bev_map, residual_map = self.model_forward(model_inputs, aperture_features, angle_features)
            final_bev_parameter = bev_map.add(residual_map)
            sync_if_cuda(self.device)
            timing.add("model_forward_s", time.perf_counter() - step_t0)

            sync_if_cuda(self.device)
            step_t0 = time.perf_counter()
            raw_scaled = prediction_to_raw_grid_cached(
                final_bev_parameter,
                patient_cache,
                ct_img,
                geometry,
                output_start,
                self.precomp,
            )
            sync_if_cuda(self.device)
            timing.add("bev_to_raw_s", time.perf_counter() - step_t0)

            sync_if_cuda(self.device)
            step_t0 = time.perf_counter()
            raw_physical = raw_scaled.clamp_min_(0.0).div_(self.dose_scale)
            body_mask = getattr(patient_cache, "body_mask_gpu", None)
            if body_mask is not None:
                if tuple(int(v) for v in body_mask.shape) != tuple(int(v) for v in raw_physical.shape):
                    raise RuntimeError(
                        "Patient body mask shape does not match predicted raw dose shape: "
                        f"{tuple(body_mask.shape)} vs {tuple(raw_physical.shape)}"
                    )
                raw_physical.masked_fill_(~body_mask, 0.0)
            if ENABLE_GPU_MINIMUM_CUTOFF and minimum_cutoff is not None:
                raw_physical.masked_fill_(raw_physical <= float(minimum_cutoff), 0.0)
            sync_if_cuda(self.device)
            timing.add("clamp_scale_s", time.perf_counter() - step_t0)

            sync_if_cuda(self.device)
            step_t0 = time.perf_counter()
            if ENABLE_PINNED_D2H and self.device.type == "cuda":
                try:
                    host = torch.empty(raw_physical.shape, dtype=torch.float32, device="cpu", pin_memory=True)
                    host.copy_(raw_physical, non_blocking=True)
                    sync_if_cuda(self.device, force=True)
                    dose_np = host.numpy()
                except RuntimeError as exc:
                    print(f"[timing] pinned_d2h status=disabled_fallback error={repr(exc)}", flush=True)
                    dose_np = raw_physical.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                dose_np = raw_physical.detach().cpu().numpy().astype(np.float32, copy=False)
            timing.add("d2h_s", time.perf_counter() - step_t0)
            return dose_np, timing.values

    def run(self) -> int:
        t0 = time.perf_counter()
        metadata_path = INPUT_PATH / f"{INPUT_JSON_NAME}.json"
        metadata_t0 = time.perf_counter()
        metadata = load_json(metadata_path)
        metadata_read_s = time.perf_counter() - metadata_t0
        if not isinstance(metadata, list):
            raise RuntimeError(f"{metadata_path} must contain a JSON array")

        output_slots: list[dict[int, tuple[np.ndarray, sitk.Image, float]]] | None = (
            None if ENABLE_ASYNC_OUTPUT_WRITER else [dict() for _ in range(NUM_OUTPUT_FILES)]
        )
        async_writer = AsyncMHAOutputWriter(metadata, OUTPUT_PATH, ASYNC_OUTPUT_QUEUE_SIZE) if ENABLE_ASYNC_OUTPUT_WRITER else None
        image_cache: dict[int, sitk.Image] = {}
        patient_cache_by_image: dict[int, object] = {}
        cp_times_s: list[float] = []
        patient_cache_times_s: list[float] = []
        cp_step_times: dict[str, list[float]] = {}
        writer_summary: dict[str, object] = {}

        try:
            for image_entry in metadata:
                image_idx = int(image_entry["image_file_idx"])
                if image_idx not in image_cache:
                    cache_t0 = time.perf_counter()
                    image_cache[image_idx] = load_input_image(image_idx)
                    patient_cache_by_image[image_idx] = build_patient_cache(
                        image_idx,
                        image_cache[image_idx],
                        self.hu,
                        self.density_curve,
                        self.device,
                    )
                    sync_if_cuda(self.device, force=True)
                    patient_cache_times_s.append(time.perf_counter() - cache_t0)
                ct_img = image_cache[image_idx]
                patient_cache = patient_cache_by_image[image_idx]

                for beam in image_entry.get("beams", []):
                    if "iso_center" not in beam and "iso_center" in image_entry:
                        beam = {**beam, "iso_center": image_entry["iso_center"]}
                    for cp in beam.get("control_points", []):
                        output_info = cp["output_info"]
                        output_file_idx = int(output_info["output_file_idx"])
                        idx_in_output = int(output_info["idx_in_output"])
                        minimum_cutoff = float(output_info["minimum_cutoff"])
                        cp_t0 = time.perf_counter()
                        dose, cp_timing = self.predict_control_point(
                            patient_cache,
                            ct_img,
                            beam,
                            cp,
                            minimum_cutoff=minimum_cutoff,
                        )
                        predict_cp_s = time.perf_counter() - cp_t0
                        cutoff_t0 = time.perf_counter()
                        if not ENABLE_GPU_MINIMUM_CUTOFF:
                            dose[dose <= minimum_cutoff] = 0.0
                        cutoff_s = time.perf_counter() - cutoff_t0
                        output_submit_s = 0.0
                        if async_writer is not None:
                            submit_t0 = time.perf_counter()
                            async_writer.submit(output_file_idx, idx_in_output, dose, ct_img)
                            output_submit_s = time.perf_counter() - submit_t0
                        else:
                            if output_slots is None:
                                raise RuntimeError("output_slots unexpectedly None")
                            output_slots[output_file_idx][idx_in_output] = (dose, ct_img, minimum_cutoff)
                        cp_total_s = predict_cp_s + cutoff_s + output_submit_s
                        cp_times_s.append(cp_total_s)
                        cp_timing["minimum_cutoff_s"] = cutoff_s
                        cp_timing["output_submit_s"] = output_submit_s
                        cp_timing["cp_total_s"] = cp_total_s
                        for key, value in cp_timing.items():
                            cp_step_times.setdefault(key, []).append(float(value))

            write_t0 = time.perf_counter()
            if async_writer is not None:
                writer_summary = async_writer.close()
            else:
                if output_slots is None:
                    raise RuntimeError("output_slots unexpectedly None")
                self.write_outputs(output_slots)
            write_outputs_s = time.perf_counter() - write_t0
        except Exception:
            if async_writer is not None:
                async_writer.abort()
            raise
        invoke_total_s = time.perf_counter() - t0
        patient_cache_mean_s = float(np.mean(patient_cache_times_s)) if patient_cache_times_s else float("nan")
        patient_cache_max_s = float(np.max(patient_cache_times_s)) if patient_cache_times_s else float("nan")
        if cp_times_s:
            cp_times = np.asarray(cp_times_s, dtype=np.float64)
            summary = {
                "metadata_read_s": metadata_read_s,
                "image_count": len(image_cache),
                "detailed_timing": int(ENABLE_DETAILED_TIMING),
                "patient_cache_mean_s": patient_cache_mean_s,
                "patient_cache_max_s": patient_cache_max_s,
                "invoke_control_points": len(cp_times_s),
                "cp_mean_s": float(cp_times.mean()),
                "cp_min_s": float(cp_times.min()),
                "cp_max_s": float(cp_times.max()),
                "write_outputs_s": write_outputs_s,
                "invoke_total_s": invoke_total_s,
                "async_output_writer": int(async_writer is not None),
                "async_output_submit_block_s": float(writer_summary.get("async_output_submit_block_s", 0.0)),
                "async_output_worker_write_s": float(writer_summary.get("async_output_worker_write_s", 0.0)),
                "async_output_bytes_mb": float(writer_summary.get("async_output_bytes_mb", 0.0)),
                "async_output_frames": int(writer_summary.get("async_output_frames", 0)),
            }
            log_timing_line("invoke_summary", summary)
            if writer_summary:
                log_timing_line(
                    "output_writer_summary",
                    {
                        "queue_size": int(ASYNC_OUTPUT_QUEUE_SIZE),
                        "writer_threads": int(ASYNC_OUTPUT_WRITER_THREADS),
                        "submit_block_s": float(writer_summary.get("async_output_submit_block_s", 0.0)),
                        "worker_write_s": float(writer_summary.get("async_output_worker_write_s", 0.0)),
                        "bytes_mb": float(writer_summary.get("async_output_bytes_mb", 0.0)),
                        "frames": int(writer_summary.get("async_output_frames", 0)),
                    },
                )
            if ENABLE_DETAILED_TIMING:
                step_summary = {}
                for key in sorted(cp_step_times):
                    values = np.asarray(cp_step_times[key], dtype=np.float64)
                    step_summary[f"{key}_mean"] = float(values.mean())
                    step_summary[f"{key}_max"] = float(values.max())
                log_timing_line("cp_breakdown", step_summary)
            else:
                log_timing_line(
                    "cp_breakdown",
                    {"detailed_timing": 0, "note": "disabled_use_ENABLE_DETAILED_TIMING_1_for_stage_times"},
                )
        else:
            log_timing_line(
                "invoke_summary",
                {
                    "metadata_read_s": metadata_read_s,
                    "image_count": len(image_cache),
                    "detailed_timing": int(ENABLE_DETAILED_TIMING),
                    "patient_cache_mean_s": patient_cache_mean_s,
                    "patient_cache_max_s": patient_cache_max_s,
                    "invoke_control_points": 0,
                    "cp_mean_s": float("nan"),
                    "cp_min_s": float("nan"),
                    "cp_max_s": float("nan"),
                    "write_outputs_s": write_outputs_s,
                    "invoke_total_s": invoke_total_s,
                    "async_output_writer": int(async_writer is not None),
                    "async_output_submit_block_s": float(writer_summary.get("async_output_submit_block_s", 0.0)),
                    "async_output_worker_write_s": float(writer_summary.get("async_output_worker_write_s", 0.0)),
                    "async_output_bytes_mb": float(writer_summary.get("async_output_bytes_mb", 0.0)),
                    "async_output_frames": int(writer_summary.get("async_output_frames", 0)),
                },
            )
        print(f"Invoke complete in {invoke_total_s:.3f}s", flush=True)
        return 0

    @staticmethod
    def write_outputs(output_slots: list[dict[int, tuple[np.ndarray, sitk.Image, float]]]) -> None:
        for output_index in range(NUM_OUTPUT_FILES):
            output_dir = OUTPUT_PATH / "images" / f"stacked-radiation-dose-map-{output_index + 1}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "output.mha"
            slot = output_slots[output_index]
            if not slot:
                write_empty_stacked_mha(output_path)
                continue

            write_stacked_mha_streaming(output_path, slot, output_index)


if __name__ == "__main__":
    raise SystemExit(DoseRadResAlgorithm().run())
