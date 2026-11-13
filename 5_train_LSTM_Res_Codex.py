#!/usr/bin/env python3
"""
Train CNN_TCN_Res_Codex.

The network input remains the existing BEV preprocessing. The network output is
a [B,Z,Y,X] BEV parameter map sampled with trilinear interpolation. The main
supervision is computed directly on the overlapping original patient-grid raw
dose voxels.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tensorboard is required only for training, not for inference.")
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent


# =============================================================================
# Hyperparameters / paths to tune directly
# =============================================================================

PREPROCESSED_ROOT = REPO_ROOT / "data" / "bev"
TRAIN_GROUP = "train"
VALIDATION_GROUP = "validation"
RAW_DOSE_ROOT = REPO_ROOT / "data" / "DoseRAD2026" / "photon" / "training"
PRECOMPUTED_GEOMETRY_NPZ = REPO_ROOT / "precomputed_geometry_v1_2" / "precompute_geometry.npz"

TRAIN_PATIENT_ID_LIST: list[str] = []
TRAIN_PATIENT_ID_RANGES: list[tuple[str, str]] = []  # Empty means use all patients in train/.

TRAIN_SAMPLE_FRACTION_PER_PATIENT = 1.0
VAL_SAMPLE_FRACTION_PER_PATIENT = 1.0
SAMPLE_SUBSET_SEED = 42

MAX_PATIENTS: int | None = None
MAX_SAMPLES: int | None = None

DEVICE = "cuda:0"
BATCH_SIZE = 1
EPOCHS = 1000
LEARNING_RATE = 1.25e-4
LR_PATIENCE = 5
NUM_WORKERS = 16
PIN_MEMORY = True
PERSISTENT_WORKERS = True

MODEL_CODEX_ROOT = REPO_ROOT / "model_Codex_Res"
RUN_NAME_SUFFIX = ""
OUTPUT_DIR: Path | None = None

SAVE_EACH_X_EPOCH: int | None = 1
SAVE_BEST_MODEL = True
USE_EMA_MODEL = True
EMA_DECAY = 0.9999
USE_GRAD_CLIP = True
GRAD_CLIP_MAX_NORM = 30.0
GRAD_CLIP_SUMMARY_EVERY = 100
MEASURE_STEP_TIMING_WITH_CUDA_SYNC = False
CONTRIBUTION_MONITOR_EVERY_STEPS = 500
TRAIN_NEGATIVE_VALUE_MONITOR_EVERY_STEPS = 100

# If RESUM_FROM_CHECKPOINT is True, LOAD_PRETRAINED_MODEL is ignored. If both are False, training starts from scratch.
LOAD_PRETRAINED_MODEL = False
PRETRAINED_CHECKPOINT: Path | None = None #(REPO_ROOT / "model_Codex" / "2026-07-19_16-29-42" / "checkpoints" / "best_model.pt")

RESUME_FROM_CHECKPOINT = False
RESUME_CHECKPOINT_PATH: Path | None = None
RESUME_RUN_ID: str | None = "2026-08-28_02-32-59"
RESUME_CHECKPOINT_EPOCH: int | None = 94
RESUME_CHECKPOINT_NAME: str | None = None
RESUME_RESTORE_OPTIMIZER = True
RESUME_RESTORE_SCHEDULER = True
RESUME_OVERRIDE_OPTIMIZER_LR = False
RESUME_RESET_BEST_VAL_LOSS = False

MODEL_ARCHITECTURE = "CNN_TCN_Res_Codex"
TCN_DILATIONS = (1, 2, 4, 8, 16, 32, 64)
TCN_KERNEL_SIZES = (
    (3, 3, 3),
    (3, 3, 3),
    
    (3, 1, 1),
    (3, 3, 3),
    (3, 1, 1),
    (3, 3, 3),
    (3, 1, 1),
)
TCN_TEMPORAL_MODES = ("symmetric", "symmetric", "causal", "causal", "causal", "causal", "causal")
TCN_DROPOUT = 0.0
TCN_NORM = "group"
DECODER_TYPE = "bilinear"
USE_DECODER_SKIPS = True
DECODER_SKIP_GATE_INIT = 0.1
USE_SKIP_Z_MIXER = True
SKIP_Z_MIXER_KERNEL = 5
USE_SKIP50 = True
USE_SKIP100 = True
ENCODER_DOWNSAMPLE_MODE = "stride_conv"  # "stride_conv" or "conv_avgpool"; affects 100->50 and 50->25 only.
OUTPUT_HEAD_MODE = "separate_standard"  # "separate_standard" or "shared_split".
USE_INPUT_CHANNEL_PADDING = True
INPUT_CT_PAD_VALUE = -1.024
INPUT_PROJ_PAD_VALUE = 0.0
INPUT_INVERSE_SQUARE_PAD_VALUE = 0.0
INPUT_WED_PAD_VALUE = 0.0

MAPS_FILENAME = "maps.h5"
CT_DATASET_NAME = "ct"
PROJ_DATASET_NAME = "proj_fraction"
DOSE_DATASET_NAME = "dose"
INVERSE_SQUARE_DATASET_NAME = "inverse_square"
WED_DATASET_NAME = "wed"

CT_SCALE = 0.001
PROJ_SCALE = 1.0
DOSE_SCALE = 10000.0
DOSE_REGION_THRESHOLD = 0.10

USE_INVERSE_SQUARE_CHANNEL = True
INVERSE_SQUARE_NORMALIZATION = "scale_offset"
INVERSE_SQUARE_SCALE = 1.0
INVERSE_SQUARE_OFFSET = -1.0
USE_WED_CHANNEL = True
WED_NORMALIZATION = "divide"
WED_SCALE_MM_WATER = 500.0

USE_ONLINE_BEV_SI_FLIP_AUGMENTATION = False
ONLINE_BEV_SI_FLIP_PROBABILITY = 0.20
ONLINE_BEV_SI_FLIP_AXIS = 1  # BEV [Z,Y,X]: Y is superior-inferior / MLC leaf-index direction.

USE_APERTURE_FILM = True
USE_ANGLE_FILM = True
APERTURE_LOG_MIN_VALUE = 1.0
APERTURE_NORMALIZED_CLAMP = 5.0
APERTURE_NORMALIZATION_TYPE = "log_train_zscore_clamp"
ANGLE_CONDITION_INPUT_TYPE = "gantry_sincos"

RAW_TRAIN_POINTS = 32768
RAW_VAL_POINTS = 100000
RAW_VAL_SEED = 20260721
RAW_HIGH_DOSE_FRACTION = 0.50
RAW_MID_DOSE_FRACTION = 0.25
RAW_UNIFORM_FRACTION = 0.25
RAW_SUPERVISION_MODE = "patient_grid_mlc_ct_regions"  # "patient_grid_mlc_ct_regions", "full_overlap", or "sampled_points"
RAW_FULL_OVERLAP_CHUNK_Z = 16
RAW_FULL_OVERLAP_BOUNDARY_EPS = 1e-4
RAW_FULL_OVERLAP_CHECK_FINITE_PER_CHUNK = False
RAW_GLOBAL_LOSS_TYPE = "mae"
RAW_DOSE_REGION_LOSS_TYPE = "mae"
RAW_GLOBAL_LOSS_WEIGHT = 0.5
RAW_DOSE_REGION_LOSS_WEIGHT = 0.5
NEGATIVE_DOSE_LOSS_WEIGHT = 0.0
BEV_AUX_LOSS_WEIGHT = 0.0
HUBER_DELTA = 0.05
USE_LEAKY_CLAMP_FOR_MAE = True
LEAKY_CLAMP_NEGATIVE_SLOPE = 0.5
NEGATIVE_VALUE_HISTOGRAM_MIN = -1.0
NEGATIVE_VALUE_HISTOGRAM_BINS = 1000

TRADITIONAL_LOSS_COMBO_NAME = "traditional_loss_combo"
LOSS_COMBO_NAME = "traditional_mlc_region_plus_dose_region_mae_plus_small_idd_combo"
DOSE_PEAK_BIN_FRACTIONS = (0.01, 0.10, 0.30, 0.80)
DOSE_PEAK_BIN_WEIGHTS = (0.5, 1.0, 1.0, 1.0, 1.0)
DOSE_PEAK_REGION_FRACTIONS = (0.01, 0.10, 0.30, 0.80)
DOSE_PEAK_REGION_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
DOSE_PEAK_GLOBAL_MAE_WEIGHT = 1.0
GRADIENT_WEIGHTED_MAE_WEIGHT = 0.0
GRADIENT_WEIGHTED_MAE_MIN_DOSE_FRACTION = 0.10
GRADIENT_WEIGHTED_MAE_EPS = 1e-8
BEV_BRANCH_LOSS_WEIGHT = 0.0
PATIENT_GRID_BRANCH_LOSS_WEIGHT = 1.8
LAMBDA_IDD_BEV = 0.0
LAMBDA_IDD_FINAL = 0.005
DETACH_BEV_IN_PATIENT_GRID_SUPERVISION = False
IDD_LOSS_EPS = 1e-12
IDD_PATIENT_TO_BEV_CHUNK_Z = 16
IDD_BODY_CT_THRESHOLD_HU = -1024.0
IDD_FILL_BODY_HOLES_2D = True
IDD_BODY_FILL_MODE = "axis_span"  # "axis_span" is fast; "iterative_flood" is exact but can hit CUDA launch timeout.
IDD_BODY_FILL_CONNECTIVITY = 4
IDD_BODY_FILL_MAX_ITERATIONS: int | None = None

USE_BEV_GRID_ANCHOR_LOSS = False
BEV_GRID_ANCHOR_LOSS_WEIGHT = 1.0
BEV_GRID_PENUMBRA_LOSS_WEIGHT = 0.5
BEV_GRID_DOSE_REGION_LOSS_WEIGHT = 0.5
BEV_GRID_PENUMBRA_MIN_FRACTION = 0.01
BEV_GRID_PENUMBRA_MAX_FRACTION = 0.10
BEV_GRID_DOSE_REGION_MIN_FRACTION = 0.10

PATIENT_GRID_GLOBAL_MAE_WEIGHT = 1.0
PATIENT_GRID_MLC_FULL_MAE_WEIGHT = 1.0
PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT = 1.0
PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT = 0.0
PATIENT_GRID_DOSE_REGION_MAE_WEIGHT = 0.0
PATIENT_GRID_DOSE_REGION_10_30_MAE_WEIGHT = 0.7
PATIENT_GRID_DOSE_REGION_30_80_MAE_WEIGHT = 0.7
PATIENT_GRID_DOSE_REGION_80_100_MAE_WEIGHT = 0.7
PATIENT_GRID_CT_BODY_THRESHOLD = -1.0
PATIENT_GRID_MLC_ZERO_EPS = 1e-4
PATIENT_GRID_MLC_FULL_EPS = 1e-4
PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION = 0.01

ENFORCE_NONNEGATIVE_BSPLINE_COEFFICIENTS = False
NONNEGATIVE_BSPLINE_METHOD = "relu"  # "relu" or "softplus"
NONNEGATIVE_SOFTPLUS_BETA = 1.0
NONNEGATIVE_SOFTPLUS_THRESHOLD = 20.0

FULL_RAW_VALIDATION_EVERY = 5
FULL_RAW_VALIDATION_MAX_SAMPLES = 1
FULL_RAW_VALIDATION_CHUNK_Z = 4
FULL_RAW_VALIDATION_COMPUTE_GAMMA = True
GAMMA_CRITERIA = ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
GAMMA_GPU_INTERP_FRACTION = 5
GAMMA_GPU_POINT_CHUNK_SIZE = 32768
GAMMA_GPU_OFFSET_CHUNK_SIZE = 128
GAMMA_GPU_SKIP_ONCE_PASSED = True

OUTPUT_REPRESENTATION = "bev_dose_plus_patient_residual"
SPATIAL_RESAMPLER = "torch_grid_sample_trilinear"
RESAMPLER_ALIGN_CORNERS = True


MODEL_REGISTRY = {
    "CNN_TCN_Res_Codex": {
        "path": REPO_ROOT / "model" / "CNN_TCN_Res_Codex" / "model.py",
        "class_name": "CNN_TCN_Res_Codex",
    },
}
SPLINE_PATH = REPO_ROOT / "model" / "CNN_TCN_Res_Codex" / "cubic_bspline.py"


def import_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SPLINE = import_module_from_path(SPLINE_PATH, "bspline_training_spline")
_GAMMA_GPU_MODULE = None


def load_gamma_gpu_module():
    global _GAMMA_GPU_MODULE
    if _GAMMA_GPU_MODULE is None:
        _GAMMA_GPU_MODULE = import_module_from_path(REPO_ROOT / "0_gamma_gpu.py", "gamma_gpu_training_bspline")
    return _GAMMA_GPU_MODULE


def gamma_metric_name(dose_percent: float, distance_mm: float) -> str:
    dose_text = f"{dose_percent:g}".replace(".", "p")
    distance_text = f"{distance_mm:g}".replace(".", "p")
    return f"full_gamma_pass_rate_{distance_text}mm_{dose_text}pct"


def model_input_channel_count() -> int:
    return 2 + int(USE_INVERSE_SQUARE_CHANNEL) + int(USE_WED_CHANNEL)


def import_class_from_path(module_path: Path, class_name: str):
    module = import_module_from_path(module_path, f"{class_name}_module")
    if not hasattr(module, class_name):
        raise ImportError(f"{module_path} does not define class {class_name}")
    return getattr(module, class_name)


def build_model() -> nn.Module:
    model_info = MODEL_REGISTRY[MODEL_ARCHITECTURE]
    model_class = import_class_from_path(model_info["path"], model_info["class_name"])
    return model_class(
        tcn_dilations=TCN_DILATIONS,
        tcn_kernel_size=TCN_KERNEL_SIZES,
        tcn_temporal_modes=TCN_TEMPORAL_MODES,
        tcn_dropout=TCN_DROPOUT,
        tcn_norm=TCN_NORM,
        decoder_type=DECODER_TYPE,
        use_decoder_skips=USE_DECODER_SKIPS,
        decoder_skip_gate_init=DECODER_SKIP_GATE_INIT,
        use_skip_z_mixer=USE_SKIP_Z_MIXER,
        skip_z_mixer_kernel=SKIP_Z_MIXER_KERNEL,
        use_skip50=USE_SKIP50,
        use_skip100=USE_SKIP100,
        encoder_downsample_mode=ENCODER_DOWNSAMPLE_MODE,
        output_head_mode=OUTPUT_HEAD_MODE,
        use_input_channel_padding=USE_INPUT_CHANNEL_PADDING,
        input_channels=model_input_channel_count(),
        input_ct_pad_value=INPUT_CT_PAD_VALUE,
        input_proj_pad_value=INPUT_PROJ_PAD_VALUE,
        input_inverse_square_pad_value=INPUT_INVERSE_SQUARE_PAD_VALUE,
        input_wed_pad_value=INPUT_WED_PAD_VALUE,
        use_aperture_film=USE_APERTURE_FILM,
        use_angle_film=USE_ANGLE_FILM,
    )


def apply_spline_output_constraints(coefficients: torch.Tensor) -> torch.Tensor:
    if not ENFORCE_NONNEGATIVE_BSPLINE_COEFFICIENTS:
        return coefficients
    method = NONNEGATIVE_BSPLINE_METHOD.lower().strip()
    if method == "relu":
        return torch.relu(coefficients)
    if method == "softplus":
        return torch.nn.functional.softplus(
            coefficients,
            beta=float(NONNEGATIVE_SOFTPLUS_BETA),
            threshold=float(NONNEGATIVE_SOFTPLUS_THRESHOLD),
        )
    raise ValueError(f"Unknown NONNEGATIVE_BSPLINE_METHOD={NONNEGATIVE_BSPLINE_METHOD!r}")


def patient_sort_key(patient_id: str) -> tuple[str, int | str]:
    match = re.match(r"^(.+?)(\d+)$", patient_id)
    if match:
        return match.group(1), int(match.group(2))
    return patient_id, patient_id


def patient_in_range(patient_id: str, start: str, end: str) -> bool:
    start_prefix, start_value = patient_sort_key(start)
    end_prefix, end_value = patient_sort_key(end)
    patient_prefix, patient_value = patient_sort_key(patient_id)
    if start_prefix == end_prefix == patient_prefix and isinstance(patient_value, int):
        return int(start_value) <= patient_value <= int(end_value)
    return start <= patient_id <= end


def train_patient_is_selected(patient_id: str) -> bool:
    if TRAIN_PATIENT_ID_LIST and patient_id in set(TRAIN_PATIENT_ID_LIST):
        return True
    if TRAIN_PATIENT_ID_RANGES:
        return any(patient_in_range(patient_id, start, end) for start, end in TRAIN_PATIENT_ID_RANGES)
    return not TRAIN_PATIENT_ID_LIST


def patient_id_from_sample_dir(sample_dir: Path) -> str:
    return sample_dir.name.split("_B", 1)[0]


def sample_is_complete(sample_dir: Path) -> bool:
    return (sample_dir / MAPS_FILENAME).exists() and (sample_dir / "metadata.json").exists()


def discover_sample_dirs(group: str, apply_train_patient_selection: bool) -> list[Path]:
    sample_dirs: list[Path] = []
    group_dir = PREPROCESSED_ROOT / group
    if not group_dir.is_dir():
        raise FileNotFoundError(group_dir)
    for sample_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        patient_id = patient_id_from_sample_dir(sample_dir)
        if apply_train_patient_selection and not train_patient_is_selected(patient_id):
            continue
        if sample_is_complete(sample_dir):
            sample_dirs.append(sample_dir)
    if not sample_dirs:
        raise RuntimeError(f"No complete samples found under {group_dir}")
    selected_patients = sorted({patient_id_from_sample_dir(p) for p in sample_dirs}, key=patient_sort_key)
    if apply_train_patient_selection and MAX_PATIENTS is not None:
        allowed = set(selected_patients[:MAX_PATIENTS])
        sample_dirs = [p for p in sample_dirs if patient_id_from_sample_dir(p) in allowed]
    return sample_dirs


def subset_samples_per_patient(samples: list[Path], fraction: float, seed: int, split_name: str) -> list[Path]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{split_name} sample fraction must be > 0 and <= 1.")
    if fraction >= 1.0:
        return samples
    by_patient: dict[str, list[Path]] = {}
    for sample in samples:
        by_patient.setdefault(patient_id_from_sample_dir(sample), []).append(sample)
    selected: list[Path] = []
    for patient_id in sorted(by_patient, key=patient_sort_key):
        patient_samples = sorted(by_patient[patient_id])
        n_keep = max(1, min(int(round(len(patient_samples) * fraction)), len(patient_samples)))
        rng = random.Random(f"{seed}:{split_name}:{patient_id}")
        selected.extend(sorted(rng.sample(patient_samples, n_keep)))
    return selected


def limit_samples_for_smoke(train_samples: list[Path], val_samples: list[Path]) -> tuple[list[Path], list[Path]]:
    if MAX_SAMPLES is None:
        return train_samples, val_samples
    if MAX_SAMPLES < 1:
        raise ValueError("MAX_SAMPLES must be at least 1.")
    return train_samples[:MAX_SAMPLES], val_samples[:MAX_SAMPLES]


def read_hdf5_dataset(path: Path, dataset_name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"{path} does not contain dataset '{dataset_name}'")
        return handle[dataset_name][()].astype(np.float32, copy=False)


def read_metadata(sample_dir: Path) -> dict:
    with open(sample_dir / "metadata.json", "r") as f:
        return json.load(f)


@dataclass(frozen=True)
class ApertureNormalizationStats:
    log_area_mean: float
    log_area_std: float
    log_s_mean: float
    log_s_std: float
    min_input_value: float = APERTURE_LOG_MIN_VALUE
    normalized_clamp: float = APERTURE_NORMALIZED_CLAMP
    normalization_type: str = APERTURE_NORMALIZATION_TYPE

    def to_dict(self) -> dict[str, float | str]:
        return {
            "aperture_normalization_type": self.normalization_type,
            "aperture_log_min_value": float(self.min_input_value),
            "aperture_normalized_clamp": float(self.normalized_clamp),
            "aperture_log_area_mean": float(self.log_area_mean),
            "aperture_log_area_std": float(self.log_area_std),
            "aperture_log_s_equivalent_mean": float(self.log_s_mean),
            "aperture_log_s_equivalent_std": float(self.log_s_std),
        }


def _numeric_values_for_key(metadata: dict, key: str) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []

    def visit(value, path: str) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_path = f"{path}.{child_key}" if path else child_key
                if child_key == key and isinstance(child_value, (int, float)):
                    found.append((child_path, float(child_value)))
                visit(child_value, child_path)
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                visit(child_value, f"{path}[{index}]")

    visit(metadata, "")
    return found


def _resolve_unique_numeric(metadata: dict, sample_key: str, keys: tuple[str, ...], field_name: str) -> float:
    checked = []
    matches: list[tuple[str, float]] = []
    for key in keys:
        checked.append(key)
        matches.extend(_numeric_values_for_key(metadata, key))
    if not matches:
        raise KeyError(
            f"Could not resolve {field_name} for {sample_key}. "
            f"Checked keys={checked}; available top-level keys={sorted(metadata.keys())}"
        )
    values = np.asarray([value for _, value in matches], dtype=np.float64)
    if not np.allclose(values, values[0], rtol=1e-7, atol=1e-8):
        raise ValueError(f"Conflicting {field_name} values for {sample_key}: {matches}")
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError(f"{field_name} is NaN/Inf for {sample_key}: {matches}")
    return value


def resolve_aperture_area_mm2(metadata: dict, sample_key: str) -> float:
    value = _resolve_unique_numeric(metadata, sample_key, ("A_open_mm2", "A_opening_mm2"), "aperture area")
    if value < 0.0:
        raise ValueError(f"Aperture area must be >= 0 for {sample_key}, got {value}")
    return value


def resolve_s_equivalent_mm(metadata: dict, sample_key: str) -> float:
    value = _resolve_unique_numeric(metadata, sample_key, ("S_equivalent_mm",), "equivalent square")
    if value < 0.0:
        raise ValueError(f"S_equivalent_mm must be >= 0 for {sample_key}, got {value}")
    return value


def resolve_gantry_angle_deg(metadata: dict, sample_key: str) -> float:
    try:
        value = float(metadata["geometry"]["gantry_angle_deg"])
    except KeyError as exc:
        raise KeyError(
            f"Could not resolve gantry angle for {sample_key}. "
            f"Expected metadata['geometry']['gantry_angle_deg']; top-level keys={sorted(metadata.keys())}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"gantry_angle_deg is NaN/Inf for {sample_key}: {value}")
    return value


def aperture_logs_from_metadata(metadata: dict, sample_key: str) -> tuple[float, float]:
    area = resolve_aperture_area_mm2(metadata, sample_key)
    s_equiv = resolve_s_equivalent_mm(metadata, sample_key)
    return (
        float(np.log(max(area, APERTURE_LOG_MIN_VALUE))),
        float(np.log(max(s_equiv, APERTURE_LOG_MIN_VALUE))),
    )


def compute_aperture_normalization_stats(train_samples: list[Path]) -> ApertureNormalizationStats:
    log_area: list[float] = []
    log_s: list[float] = []
    for sample_dir in train_samples:
        area_value, s_value = aperture_logs_from_metadata(read_metadata(sample_dir), sample_dir.name)
        log_area.append(area_value)
        log_s.append(s_value)
    area_arr = np.asarray(log_area, dtype=np.float64)
    s_arr = np.asarray(log_s, dtype=np.float64)
    area_std = float(np.std(area_arr, ddof=0))
    s_std = float(np.std(s_arr, ddof=0))
    if area_std < 1e-8 or s_std < 1e-8:
        raise RuntimeError(f"Aperture normalization std too small: area_std={area_std}, s_std={s_std}")
    return ApertureNormalizationStats(
        log_area_mean=float(np.mean(area_arr)),
        log_area_std=area_std,
        log_s_mean=float(np.mean(s_arr)),
        log_s_std=s_std,
    )


def normalize_aperture_features(area_mm2: float, s_equivalent_mm: float, stats: ApertureNormalizationStats) -> np.ndarray:
    log_area = np.log(max(float(area_mm2), stats.min_input_value))
    log_s = np.log(max(float(s_equivalent_mm), stats.min_input_value))
    features = np.asarray(
        [
            (log_area - stats.log_area_mean) / stats.log_area_std,
            (log_s - stats.log_s_mean) / stats.log_s_std,
        ],
        dtype=np.float32,
    )
    return np.clip(features, -float(stats.normalized_clamp), float(stats.normalized_clamp)).astype(np.float32, copy=False)


def gantry_angle_features(gantry_angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(gantry_angle_deg))
    return np.asarray([np.sin(theta), np.cos(theta)], dtype=np.float32)


def load_preprocessing_module():
    module_path = REPO_ROOT / "1_Preprocessing.py"
    return import_module_from_path(module_path, "raw_preprocessing_discovery")


def build_raw_task_lookup(sample_dirs: list[Path]) -> dict[str, dict[str, object]]:
    pre = load_preprocessing_module()
    selected_patient_ids = {patient_id_from_sample_dir(p) for p in sample_dirs}
    tasks = pre.discover_samples(RAW_DOSE_ROOT, selected_patient_ids, None, None, 0.0, None)
    lookup: dict[str, dict[str, object]] = {}
    for task in tasks:
        geometry = pre.geometry_from_beam_cp(task.beam, task.cp)
        lookup[task.key] = {
            "sample_key": task.key,
            "patient_id": task.patient_id,
            "raw_dose_path": str(task.dose_path),
            "geometry": asdict(geometry),
        }
    missing = sorted(p.name for p in sample_dirs if p.name not in lookup)
    if missing:
        raise KeyError(f"Raw task lookup is missing {len(missing)} samples; first missing: {missing[:5]}")
    return lookup


def inverse_square_from_metadata(metadata: dict) -> np.ndarray:
    depth, height, width = (int(v) for v in metadata["shape_depth_si_lateral"])
    spacing = float(metadata.get("spacing_mm", 2.0))
    z0 = float(metadata["output_z_min_mm"])
    z = z0 + np.arange(depth, dtype=np.float32)[:, None, None] * spacing
    y = (np.arange(height, dtype=np.float32)[None, :, None] - (height - 1) / 2.0) * spacing
    x = (np.arange(width, dtype=np.float32)[None, None, :] - (width - 1) / 2.0) * spacing
    return (1000.0 / np.sqrt(x * x + y * y + (z + 1000.0) * (z + 1000.0))).astype(np.float32) ** 2


def normalize_inverse_square(inverse_square: np.ndarray) -> np.ndarray:
    if INVERSE_SQUARE_NORMALIZATION == "scale_offset":
        return inverse_square * INVERSE_SQUARE_SCALE + INVERSE_SQUARE_OFFSET
    if INVERSE_SQUARE_NORMALIZATION == "none":
        return inverse_square
    raise ValueError(f"Unknown INVERSE_SQUARE_NORMALIZATION={INVERSE_SQUARE_NORMALIZATION!r}")


def normalize_wed(wed: np.ndarray) -> np.ndarray:
    if WED_NORMALIZATION == "divide":
        return wed / WED_SCALE_MM_WATER
    if WED_NORMALIZATION == "none":
        return wed
    raise ValueError(f"Unknown WED_NORMALIZATION={WED_NORMALIZATION!r}")


def load_precomputed_geometry() -> dict[str, np.ndarray]:
    if not PRECOMPUTED_GEOMETRY_NPZ.exists():
        raise FileNotFoundError(PRECOMPUTED_GEOMETRY_NPZ)
    return dict(np.load(PRECOMPUTED_GEOMETRY_NPZ))


def raw_index_to_bev_affine(raw_img: sitk.Image, metadata: dict, precomp: dict[str, np.ndarray]) -> np.ndarray:
    geometry = metadata["geometry"]
    iso = np.asarray(geometry["iso_center_xyz"], dtype=np.float64)
    lateral = np.asarray(geometry["lateral_axis_xyz"], dtype=np.float64)
    si = np.asarray(geometry["si_axis_xyz"], dtype=np.float64)
    beam = np.asarray(geometry["beam_axis_xyz"], dtype=np.float64)
    basis = np.stack((lateral, si, beam), axis=0)

    origin = np.asarray(raw_img.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(raw_img.GetSpacing(), dtype=np.float64)
    direction = np.asarray(raw_img.GetDirection(), dtype=np.float64).reshape(3, 3)
    image_index_to_world = direction @ np.diag(spacing)

    x_centers = precomp["cart_x_centers_mm"].astype(np.float64)
    y_centers = precomp["cart_y_centers_mm"].astype(np.float64)
    z_master = precomp["master_z_centers_mm"].astype(np.float64)
    output_start = int(metadata["output_start_index"])
    base_mm = np.asarray(
        [x_centers[0], y_centers[0], z_master[output_start]],
        dtype=np.float64,
    )
    bev_spacing = np.asarray(
        [
            float(x_centers[1] - x_centers[0]),
            float(y_centers[1] - y_centers[0]),
            float(z_master[1] - z_master[0]),
        ],
        dtype=np.float64,
    )
    index_linear = np.diag(1.0 / bev_spacing) @ basis @ image_index_to_world
    index_offset = (basis @ (origin - iso) - base_mm) / bev_spacing
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = index_linear.astype(np.float32)
    affine[:3, 3] = index_offset.astype(np.float32)
    return affine


def flip_raw_to_bev_affine_si_axis(affine_xyz: np.ndarray, bev_height: int) -> np.ndarray:
    """Compose raw->BEV affine with BEV Y-index flip for online SI augmentation."""
    if affine_xyz.shape != (4, 4):
        raise ValueError(f"affine_xyz must be [4,4], got {affine_xyz.shape}")
    flipped = np.array(affine_xyz, dtype=np.float32, copy=True)
    flipped[1, :3] *= -1.0
    flipped[1, 3] = float(bev_height - 1) - flipped[1, 3]
    return flipped


def maybe_apply_online_bev_si_flip(
    input_tensors: list[torch.Tensor],
    dose_bev: torch.Tensor,
    affine: np.ndarray,
    probability: float,
) -> tuple[list[torch.Tensor], torch.Tensor, np.ndarray, bool]:
    if probability <= 0.0 or random.random() >= probability:
        return input_tensors, dose_bev, affine, False
    if not input_tensors:
        raise ValueError("input_tensors must not be empty.")
    height = int(input_tensors[0].shape[1])
    flipped_inputs = [torch.flip(tensor, dims=(ONLINE_BEV_SI_FLIP_AXIS,)).contiguous() for tensor in input_tensors]
    flipped_dose = torch.flip(dose_bev, dims=(ONLINE_BEV_SI_FLIP_AXIS,)).contiguous()
    flipped_affine = flip_raw_to_bev_affine_si_axis(affine, height)
    return flipped_inputs, flipped_dose, flipped_affine, True


class BSplineRawDoseDataset(Dataset):
    def __init__(
        self,
        sample_dirs: list[Path],
        raw_lookup: dict[str, dict[str, object]],
        precomp: dict[str, np.ndarray],
        aperture_stats: ApertureNormalizationStats,
        enable_augmentation: bool = False,
    ):
        self.sample_dirs = sample_dirs
        self.raw_lookup = raw_lookup
        self.precomp = precomp
        self.aperture_stats = aperture_stats
        self.enable_augmentation = bool(enable_augmentation)
        if not self.sample_dirs:
            raise RuntimeError("Empty dataset.")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_dir = self.sample_dirs[idx]
        metadata = read_metadata(sample_dir)
        sample_key = sample_dir.name
        a_open_mm2 = resolve_aperture_area_mm2(metadata, sample_key)
        s_equivalent_mm = resolve_s_equivalent_mm(metadata, sample_key)
        gantry_angle_deg = resolve_gantry_angle_deg(metadata, sample_key)
        aperture_features = normalize_aperture_features(a_open_mm2, s_equivalent_mm, self.aperture_stats)
        angle_features = gantry_angle_features(gantry_angle_deg)
        raw_info = self.raw_lookup[sample_key]
        raw_dose_path = Path(str(raw_info["raw_dose_path"]))
        metadata_dose_path = Path(str(metadata.get("dose_path", raw_dose_path)))
        if metadata_dose_path.resolve() != raw_dose_path.resolve():
            raise RuntimeError(f"Metadata dose path and raw lookup disagree for {sample_key}")

        maps_path = sample_dir / MAPS_FILENAME
        ct = read_hdf5_dataset(maps_path, CT_DATASET_NAME) * CT_SCALE
        proj = read_hdf5_dataset(maps_path, PROJ_DATASET_NAME) * PROJ_SCALE
        input_tensors: list[torch.Tensor] = [torch.from_numpy(ct), torch.from_numpy(proj)]
        if USE_INVERSE_SQUARE_CHANNEL:
            input_tensors.append(torch.from_numpy(normalize_inverse_square(inverse_square_from_metadata(metadata))))
        if USE_WED_CHANNEL:
            input_tensors.append(torch.from_numpy(normalize_wed(read_hdf5_dataset(maps_path, WED_DATASET_NAME))))

        dose_bev = read_hdf5_dataset(maps_path, DOSE_DATASET_NAME) * DOSE_SCALE
        raw_img = sitk.ReadImage(str(raw_dose_path), sitk.sitkFloat32)
        raw_dose = sitk.GetArrayFromImage(raw_img).astype(np.float32, copy=False) * DOSE_SCALE
        affine = raw_index_to_bev_affine(raw_img, metadata, self.precomp)
        augmented = False
        if self.enable_augmentation and USE_ONLINE_BEV_SI_FLIP_AUGMENTATION:
            input_tensors, dose_bev_t, affine, augmented = maybe_apply_online_bev_si_flip(
                input_tensors,
                torch.from_numpy(dose_bev),
                affine,
                float(ONLINE_BEV_SI_FLIP_PROBABILITY),
            )
        else:
            dose_bev_t = torch.from_numpy(dose_bev)

        return {
            "inputs": tuple(input_tensors),
            "aperture_features": torch.from_numpy(aperture_features),
            "angle_features": torch.from_numpy(angle_features),
            "a_open_mm2": float(a_open_mm2),
            "s_equivalent_mm": float(s_equivalent_mm),
            "gantry_angle_deg": float(gantry_angle_deg),
            "dose_bev": dose_bev_t,
            "dose_raw": torch.from_numpy(raw_dose),
            "raw_index_to_bev": torch.from_numpy(affine),
            "raw_shape_zyx": tuple(int(v) for v in raw_dose.shape),
            "raw_origin_xyz": tuple(float(v) for v in raw_img.GetOrigin()),
            "raw_spacing_xyz": tuple(float(v) for v in raw_img.GetSpacing()),
            "raw_direction": tuple(float(v) for v in raw_img.GetDirection()),
            "sample_key": sample_key,
            "sample_dir": str(sample_dir),
            "raw_dose_path": str(raw_dose_path),
            "bev_si_flip_augmented": augmented,
        }


def ragged_raw_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Empty batch.")
    input_count = len(batch[0]["inputs"])  # type: ignore[arg-type]
    for item in batch:
        if len(item["inputs"]) != input_count:  # type: ignore[arg-type]
            raise ValueError("All samples in a batch must have the same number of BEV input channels.")
    return {
        "inputs": tuple(
            torch.stack([item["inputs"][channel_idx] for item in batch], dim=0)  # type: ignore[index]
            for channel_idx in range(input_count)
        ),
        "aperture_features": torch.stack([item["aperture_features"] for item in batch], dim=0),  # type: ignore[list-item]
        "angle_features": torch.stack([item["angle_features"] for item in batch], dim=0),  # type: ignore[list-item]
        "a_open_mm2": [float(item["a_open_mm2"]) for item in batch],
        "s_equivalent_mm": [float(item["s_equivalent_mm"]) for item in batch],
        "gantry_angle_deg": [float(item["gantry_angle_deg"]) for item in batch],
        "dose_bev": torch.stack([item["dose_bev"] for item in batch], dim=0),  # type: ignore[list-item]
        "dose_raw": [item["dose_raw"] for item in batch],
        "raw_index_to_bev": [item["raw_index_to_bev"] for item in batch],
        "raw_shape_zyx": [item["raw_shape_zyx"] for item in batch],
        "raw_origin_xyz": [item["raw_origin_xyz"] for item in batch],
        "raw_spacing_xyz": [item["raw_spacing_xyz"] for item in batch],
        "raw_direction": [item["raw_direction"] for item in batch],
        "sample_key": [str(item["sample_key"]) for item in batch],
        "sample_dir": [str(item["sample_dir"]) for item in batch],
        "raw_dose_path": [str(item["raw_dose_path"]) for item in batch],
        "bev_si_flip_augmented": [bool(item.get("bev_si_flip_augmented", False)) for item in batch],
    }


def ensure_batched(batch: dict[str, object]) -> dict[str, object]:
    if isinstance(batch["sample_key"], list):
        return batch
    return ragged_raw_collate([batch])


def validate_loss_type(loss_type: str) -> str:
    normalized = loss_type.lower().strip()
    if normalized == "mae":
        normalized = "l1"
    valid = {"mse", "l1", "huber", "none"}
    if normalized not in valid:
        raise ValueError(f"Unknown loss type {loss_type!r}. Valid options: {sorted(valid)}")
    return normalized


def pointwise_loss(diff: torch.Tensor, loss_type: str) -> torch.Tensor:
    loss_type = validate_loss_type(loss_type)
    if loss_type == "mse":
        return diff * diff
    if loss_type == "l1":
        return torch.abs(diff)
    if loss_type == "huber":
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * diff * diff
        linear = HUBER_DELTA * (abs_diff - 0.5 * HUBER_DELTA)
        return torch.where(abs_diff <= HUBER_DELTA, quadratic, linear)
    raise ValueError("pointwise_loss should not be called with loss_type='none'.")


def reduced_loss(prediction: torch.Tensor, reference: torch.Tensor, loss_type: str, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss_type = validate_loss_type(loss_type)
    if loss_type == "none":
        return prediction.sum() * 0.0
    diff = prediction - reference
    if mask is not None:
        if not torch.any(mask):
            return prediction.sum() * 0.0
        diff = diff[mask]
    return torch.mean(pointwise_loss(diff, loss_type))


def masked_l1_mean_no_sync(diff: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.to(dtype=diff.dtype)
    count = torch.sum(mask_f)
    value = torch.sum(torch.abs(diff) * mask_f) / torch.clamp(count, min=1.0)
    return value, count


def leaky_clamp_prediction_for_mae(prediction: torch.Tensor) -> torch.Tensor:
    if not bool(USE_LEAKY_CLAMP_FOR_MAE):
        return prediction
    slope = float(LEAKY_CLAMP_NEGATIVE_SLOPE)
    if slope == 1.0:
        return prediction
    return torch.where(prediction < 0, prediction * slope, prediction)


def negative_value_histogram_update(
    hist: torch.Tensor,
    prediction: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neg = valid & (prediction < 0)
    neg_f = neg.to(dtype=hist.dtype)
    neg_count = torch.sum(neg_f)
    pred_for_bins = torch.clamp(
        prediction.detach(),
        min=float(NEGATIVE_VALUE_HISTOGRAM_MIN),
        max=0.0,
    )
    bins = int(NEGATIVE_VALUE_HISTOGRAM_BINS)
    span = abs(float(NEGATIVE_VALUE_HISTOGRAM_MIN))
    bucket = torch.floor((pred_for_bins - float(NEGATIVE_VALUE_HISTOGRAM_MIN)) / span * bins).to(dtype=torch.long)
    bucket = torch.clamp(bucket, min=0, max=bins - 1)
    hist = hist.scatter_add(0, bucket.reshape(-1), neg_f.reshape(-1))
    neg_min = torch.min(torch.where(neg, prediction.detach(), torch.full_like(prediction, float("inf"))))
    return hist, neg_count, neg_min


def negative_value_histogram_metrics(
    hist: torch.Tensor,
    count: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    count_value = float(count.detach().cpu())
    if count_value <= 0:
        return {
            f"{prefix}_negative_value_p05": float("nan"),
            f"{prefix}_negative_value_p95": float("nan"),
        }
    hist_cpu = hist.detach().cpu()
    cdf = torch.cumsum(hist_cpu, dim=0)
    bins = int(NEGATIVE_VALUE_HISTOGRAM_BINS)
    width = abs(float(NEGATIVE_VALUE_HISTOGRAM_MIN)) / float(bins)

    def value_at_fraction(fraction: float) -> float:
        idx = int(torch.searchsorted(cdf, torch.tensor(count_value * fraction, dtype=cdf.dtype)).item())
        idx = max(0, min(idx, bins - 1))
        return float(NEGATIVE_VALUE_HISTOGRAM_MIN) + (idx + 0.5) * width

    return {
        f"{prefix}_negative_value_p05": value_at_fraction(0.05),
        f"{prefix}_negative_value_p95": value_at_fraction(0.95),
    }


def dose_peak_bin_weight_map(reference: torch.Tensor, ref_peak: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if len(DOSE_PEAK_BIN_WEIGHTS) != len(DOSE_PEAK_BIN_FRACTIONS) + 1:
        raise ValueError(
            "DOSE_PEAK_BIN_WEIGHTS must have exactly len(DOSE_PEAK_BIN_FRACTIONS)+1 entries, "
            f"got {len(DOSE_PEAK_BIN_WEIGHTS)} vs fractions={len(DOSE_PEAK_BIN_FRACTIONS)}"
        )
    ref = reference.to(dtype=torch.float32)
    peak = torch.clamp(ref_peak.to(device=reference.device, dtype=torch.float32), min=1e-12)
    fractions = [float(value) for value in DOSE_PEAK_BIN_FRACTIONS]
    weights = [float(value) for value in DOSE_PEAK_BIN_WEIGHTS]
    bins = torch.full_like(ref, weights[-1], dtype=torch.float32)
    for bin_idx in range(len(fractions) - 1, -1, -1):
        bins = torch.where(ref < fractions[bin_idx] * peak, torch.full_like(bins, weights[bin_idx]), bins)
    if valid_mask is not None:
        bins = bins * valid_mask.to(device=bins.device, dtype=bins.dtype)
    return bins.to(dtype=reference.dtype)


def dose_peak_bin_weighted_l1_mean(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    ref_peak: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    diff_abs = torch.abs(prediction - reference)
    weights = dose_peak_bin_weight_map(reference, ref_peak, valid_mask=valid_mask)
    weight_sum = torch.sum(weights)
    loss = torch.sum(diff_abs * weights) / torch.clamp(weight_sum, min=1.0)
    return loss, weight_sum


def dose_peak_region_masks(reference: torch.Tensor, ref_peak: torch.Tensor) -> list[torch.Tensor]:
    if len(DOSE_PEAK_REGION_FRACTIONS) != 4:
        raise ValueError(f"DOSE_PEAK_REGION_FRACTIONS must define 4 thresholds, got {DOSE_PEAK_REGION_FRACTIONS}")
    f01, f10, f30, f80 = [float(value) for value in DOSE_PEAK_REGION_FRACTIONS]
    peak = torch.clamp(ref_peak.to(device=reference.device, dtype=reference.dtype), min=1e-12)
    return [
        (reference >= f01 * peak) & (reference < f10 * peak),
        (reference >= f10 * peak) & (reference < f30 * peak),
        (reference >= f30 * peak) & (reference < f80 * peak),
        reference >= f80 * peak,
    ]


def dose_peak_region_balanced_l1_mean(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    ref_peak: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != reference.shape:
        raise ValueError(f"Region-balanced MAE shape mismatch: {tuple(prediction.shape)} vs {tuple(reference.shape)}")
    diff = prediction - reference
    if valid_mask is None:
        valid = torch.ones_like(reference, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=reference.device, dtype=torch.bool)
    global_mae, global_count = masked_l1_mean_no_sync(diff, valid)
    region_masks = [mask & valid for mask in dose_peak_region_masks(reference, ref_peak)]
    region_maes_counts = [masked_l1_mean_no_sync(diff, mask) for mask in region_masks]
    region_maes = [item[0] for item in region_maes_counts]
    region_counts = [item[1] for item in region_maes_counts]
    region_weights = [float(value) for value in DOSE_PEAK_REGION_WEIGHTS]
    if len(region_weights) != len(region_maes):
        raise ValueError(
            "DOSE_PEAK_REGION_WEIGHTS must have the same length as the dose peak regions, "
            f"got {len(region_weights)} vs {len(region_maes)}"
        )
    total = float(DOSE_PEAK_GLOBAL_MAE_WEIGHT) * global_mae
    for weight, region_mae in zip(region_weights, region_maes):
        total = total + weight * region_mae
    metrics = {
        "global_mae": global_mae,
        "global_count": global_count,
        "region_1_10_mae": region_maes[0],
        "region_1_10_count": region_counts[0],
        "region_10_30_mae": region_maes[1],
        "region_10_30_count": region_counts[1],
        "region_30_80_mae": region_maes[2],
        "region_30_80_count": region_counts[2],
        "region_80_plus_mae": region_maes[3],
        "region_80_plus_count": region_counts[3],
    }
    return total, metrics


def dose_gradient_magnitude_3d(reference_zyx: torch.Tensor) -> torch.Tensor:
    if reference_zyx.ndim != 3:
        raise ValueError(f"reference_zyx must be [Z,Y,X], got {tuple(reference_zyx.shape)}")
    ref = reference_zyx
    dz = torch.zeros_like(ref)
    dy = torch.zeros_like(ref)
    dx = torch.zeros_like(ref)
    dz[1:, :, :] = ref[1:, :, :] - ref[:-1, :, :]
    dy[:, 1:, :] = ref[:, 1:, :] - ref[:, :-1, :]
    dx[:, :, 1:] = ref[:, :, 1:] - ref[:, :, :-1]
    return torch.sqrt(dx * dx + dy * dy + dz * dz)


def gradient_weighted_mae_loss_3d(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    ref_peak: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    reference_gradient: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != reference.shape:
        raise ValueError(f"Gradient MAE shape mismatch: {tuple(prediction.shape)} vs {tuple(reference.shape)}")
    grad = dose_gradient_magnitude_3d(reference) if reference_gradient is None else reference_gradient.to(device=reference.device, dtype=reference.dtype)
    if grad.shape != reference.shape:
        raise ValueError(f"reference_gradient shape mismatch: {tuple(grad.shape)} vs {tuple(reference.shape)}")
    peak = torch.clamp(ref_peak.to(device=reference.device, dtype=reference.dtype), min=1e-12)
    mask = reference > float(GRADIENT_WEIGHTED_MAE_MIN_DOSE_FRACTION) * peak
    weight = grad * mask.to(dtype=reference.dtype)
    if valid_mask is not None:
        weight = weight * valid_mask.to(device=reference.device, dtype=reference.dtype)
    weight_sum = torch.sum(weight)
    loss = torch.sum(torch.abs(prediction - reference) * weight) / (weight_sum + float(GRADIENT_WEIGHTED_MAE_EPS))
    return loss, weight_sum


def compute_bev_grid_anchor_loss_and_metrics(
    parameter_map: torch.Tensor,
    dose_bev: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if parameter_map.ndim != 4 or parameter_map.shape[0] != 1:
        raise ValueError(f"parameter_map must be [1,Z,Y,X], got {tuple(parameter_map.shape)}")
    if dose_bev.ndim == 4 and dose_bev.shape[0] == 1:
        dose_bev = dose_bev[0]
    if dose_bev.ndim != 3:
        raise ValueError(f"dose_bev must be [Z,Y,X], got {tuple(dose_bev.shape)}")
    pred_bev = parameter_map[0]
    ref_bev = dose_bev.to(device=pred_bev.device, dtype=pred_bev.dtype)
    if pred_bev.shape != ref_bev.shape:
        raise ValueError(f"BEV prediction/reference shape mismatch: {tuple(pred_bev.shape)} vs {tuple(ref_bev.shape)}")

    ref_peak = torch.clamp(torch.max(ref_bev), min=1e-12)
    penumbra_mask = (
        (ref_bev >= float(BEV_GRID_PENUMBRA_MIN_FRACTION) * ref_peak)
        & (ref_bev < float(BEV_GRID_PENUMBRA_MAX_FRACTION) * ref_peak)
    )
    dose_region_mask = ref_bev >= float(BEV_GRID_DOSE_REGION_MIN_FRACTION) * ref_peak
    diff = pred_bev - ref_bev
    penumbra_mae, penumbra_count = masked_l1_mean_no_sync(diff, penumbra_mask)
    dose_region_mae, dose_region_count = masked_l1_mean_no_sync(diff, dose_region_mask)
    anchor_loss = (
        float(BEV_GRID_PENUMBRA_LOSS_WEIGHT) * penumbra_mae
        + float(BEV_GRID_DOSE_REGION_LOSS_WEIGHT) * dose_region_mae
    )
    if not USE_BEV_GRID_ANCHOR_LOSS:
        anchor_loss = anchor_loss * 0.0

    with torch.no_grad():
        metrics = {
            "bev_grid_anchor_loss": float(anchor_loss.detach().cpu()),
            "bev_grid_penumbra_mae": float(penumbra_mae.detach().cpu()),
            "bev_grid_dose_region_mae": float(dose_region_mae.detach().cpu()),
        }
    return float(BEV_GRID_ANCHOR_LOSS_WEIGHT) * anchor_loss, metrics


def _sample_indices_from_candidates(candidates: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    if n <= 0:
        return candidates.new_empty((0, 3))
    if candidates.numel() == 0:
        return candidates.new_empty((0, 3))
    choice = torch.randint(candidates.shape[0], (n,), device=candidates.device, generator=generator)
    return candidates[choice]


def sample_raw_training_points(dose_raw: torch.Tensor, n_points: int, generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        generator = torch.Generator(device=dose_raw.device)
        generator.seed()
    ref_max = torch.max(dose_raw)
    if not torch.isfinite(ref_max):
        raise RuntimeError("raw dose max is NaN/Inf")
    high_mask = dose_raw >= DOSE_REGION_THRESHOLD * ref_max
    mid_mask = (dose_raw >= 0.01 * ref_max) & (dose_raw < DOSE_REGION_THRESHOLD * ref_max)
    high = torch.nonzero(high_mask, as_tuple=False)
    mid = torch.nonzero(mid_mask, as_tuple=False)

    n_high = int(round(n_points * RAW_HIGH_DOSE_FRACTION))
    n_mid = int(round(n_points * RAW_MID_DOSE_FRACTION))
    n_uniform = max(0, n_points - n_high - n_mid)
    selected = [
        _sample_indices_from_candidates(high, n_high, generator),
        _sample_indices_from_candidates(mid, n_mid, generator),
    ]

    raw_z, raw_y, raw_x = dose_raw.shape
    uniform_zyx = torch.stack(
        (
            torch.randint(raw_z, (n_uniform,), device=dose_raw.device, generator=generator),
            torch.randint(raw_y, (n_uniform,), device=dose_raw.device, generator=generator),
            torch.randint(raw_x, (n_uniform,), device=dose_raw.device, generator=generator),
        ),
        dim=1,
    )
    selected.append(uniform_zyx)
    indices = torch.cat([item for item in selected if item.numel() > 0], dim=0)
    if indices.shape[0] < n_points:
        extra = torch.stack(
            (
                torch.randint(raw_z, (n_points - indices.shape[0],), device=dose_raw.device, generator=generator),
                torch.randint(raw_y, (n_points - indices.shape[0],), device=dose_raw.device, generator=generator),
                torch.randint(raw_x, (n_points - indices.shape[0],), device=dose_raw.device, generator=generator),
            ),
            dim=1,
        )
        indices = torch.cat((indices, extra), dim=0)
    if indices.shape[0] > n_points:
        indices = indices[:n_points]
    return indices


def compute_losses_and_metrics(
    parameter_map: torch.Tensor,
    dose_raw: torch.Tensor,
    dose_bev: torch.Tensor,
    affine: torch.Tensor,
    raw_indices_zyx: torch.Tensor,
    spline_time_s: float,
    forward_time_s: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    coords_bev_xyz = SPLINE.raw_indices_zyx_to_bev_xyz(raw_indices_zyx, affine)
    valid = valid_bev_overlap_mask(coords_bev_xyz, tuple(int(v) for v in parameter_map.shape[1:]))
    coords_bev_xyz = coords_bev_xyz[valid]
    raw_indices_zyx = raw_indices_zyx[valid]
    if raw_indices_zyx.shape[0] == 0:
        raise RuntimeError("No sampled raw dose points overlap the BEV output grid.")
    pred_raw_points = sample_trilinear_parameter_map_points(parameter_map, coords_bev_xyz)
    label_raw_points = dose_raw[raw_indices_zyx[:, 0], raw_indices_zyx[:, 1], raw_indices_zyx[:, 2]]

    if not torch.isfinite(pred_raw_points).all() or not torch.isfinite(label_raw_points).all():
        raise RuntimeError("NaN/Inf encountered in raw sampled predictions or labels.")

    ref_max = torch.clamp(torch.max(dose_raw), min=1e-12)
    dose_region = label_raw_points >= DOSE_REGION_THRESHOLD * ref_max
    global_loss = reduced_loss(pred_raw_points, label_raw_points, RAW_GLOBAL_LOSS_TYPE)
    dose_region_loss = reduced_loss(
        pred_raw_points,
        label_raw_points,
        RAW_DOSE_REGION_LOSS_TYPE,
        mask=dose_region,
    )
    raw_loss = RAW_GLOBAL_LOSS_WEIGHT * global_loss + RAW_DOSE_REGION_LOSS_WEIGHT * dose_region_loss
    negative_loss = torch.mean(torch.relu(-pred_raw_points) ** 2)
    bev_aux_loss = parameter_map.sum() * 0.0
    if BEV_AUX_LOSS_WEIGHT != 0.0:
        bev_aux_loss = reduced_loss(parameter_map[0], dose_bev, RAW_GLOBAL_LOSS_TYPE)
    bev_grid_anchor_loss, bev_grid_metrics = compute_bev_grid_anchor_loss_and_metrics(parameter_map, dose_bev)
    total_loss = (
        raw_loss
        + bev_grid_anchor_loss
        + NEGATIVE_DOSE_LOSS_WEIGHT * negative_loss
        + BEV_AUX_LOSS_WEIGHT * bev_aux_loss
    )

    with torch.no_grad():
        diff = pred_raw_points - label_raw_points
        abs_diff = torch.abs(diff)
        mid = (label_raw_points >= 0.01 * ref_max) & (label_raw_points < DOSE_REGION_THRESHOLD * ref_max)
        param = parameter_map.detach()
        global_mse = torch.mean(diff * diff)
        global_mae = torch.mean(abs_diff)
        if torch.any(dose_region):
            dose_region_diff = diff[dose_region]
            dose_region_abs_diff = abs_diff[dose_region]
            dose_region_mse = torch.mean(dose_region_diff * dose_region_diff)
            dose_region_mae = torch.mean(dose_region_abs_diff)
            dose_region_point_fraction = dose_region.float().mean()
        else:
            dose_region_mse = torch.zeros((), device=diff.device)
            dose_region_mae = torch.zeros((), device=diff.device)
            dose_region_point_fraction = torch.zeros((), device=diff.device)
        metrics = {
            "global_mse": float(global_mse.detach().cpu()),
            "global_mae": float(global_mae.detach().cpu()),
            "dose_region_mse": float(dose_region_mse.detach().cpu()),
            "dose_region_mae": float(dose_region_mae.detach().cpu()),
            "dose_region_point_fraction": float(dose_region_point_fraction.detach().cpu()),
            "raw_point_mse": float(global_mse.detach().cpu()),
            "raw_point_mae": float(global_mae.detach().cpu()),
            "raw_high_region_mae": float(dose_region_mae.detach().cpu()) if torch.any(dose_region) else float("nan"),
            "raw_mid_region_mae": float(torch.mean(abs_diff[mid]).detach().cpu()) if torch.any(mid) else float("nan"),
            "negative_prediction_fraction": float((pred_raw_points < 0).float().mean().detach().cpu()),
            "sampled_prediction_min": float(torch.min(pred_raw_points).detach().cpu()),
            "sampled_prediction_max": float(torch.max(pred_raw_points).detach().cpu()),
            "prediction_reference_max_ratio": float((torch.max(pred_raw_points) / ref_max).detach().cpu()),
            "coefficient_min": float(torch.min(param).cpu()),
            "coefficient_max": float(torch.max(param).cpu()),
            "coefficient_mean": float(torch.mean(param).cpu()),
            "coefficient_std": float(torch.std(param).cpu()),
            "parameter_min": float(torch.min(param).cpu()),
            "parameter_max": float(torch.max(param).cpu()),
            "parameter_mean": float(torch.mean(param).cpu()),
            "parameter_std": float(torch.std(param).cpu()),
            "global_loss": float(global_loss.detach().cpu()),
            "dose_region_loss": float(dose_region_loss.detach().cpu()),
            "raw_loss": float(raw_loss.detach().cpu()),
            "negative_loss": float(negative_loss.detach().cpu()),
            "bev_aux_loss": float(bev_aux_loss.detach().cpu()),
            **bev_grid_metrics,
            "spline_sampling_time_s": float(spline_time_s),
            "model_forward_time_s": float(forward_time_s),
        }
    return total_loss, metrics


def raw_overlap_chunk_indices(
    z0: int,
    z1: int,
    raw_y: int,
    raw_x: int,
    device: torch.device,
) -> torch.Tensor:
    z = torch.arange(z0, z1, dtype=torch.float32, device=device)
    y = torch.arange(raw_y, dtype=torch.float32, device=device)
    x = torch.arange(raw_x, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)


def valid_bev_overlap_mask(coords_bev_xyz: torch.Tensor, parameter_shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = parameter_shape_zyx
    eps = float(RAW_FULL_OVERLAP_BOUNDARY_EPS)
    return (
        (coords_bev_xyz[:, 0] >= -eps)
        & (coords_bev_xyz[:, 0] <= float(width - 1) + eps)
        & (coords_bev_xyz[:, 1] >= -eps)
        & (coords_bev_xyz[:, 1] <= float(height - 1) + eps)
        & (coords_bev_xyz[:, 2] >= -eps)
        & (coords_bev_xyz[:, 2] <= float(depth - 1) + eps)
    )


def bev_xyz_to_grid_sample(coords_bev_xyz: torch.Tensor, parameter_shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = parameter_shape_zyx
    x_norm = coords_bev_xyz[:, 0] * (2.0 / float(max(width - 1, 1))) - 1.0
    y_norm = coords_bev_xyz[:, 1] * (2.0 / float(max(height - 1, 1))) - 1.0
    z_norm = coords_bev_xyz[:, 2] * (2.0 / float(max(depth - 1, 1))) - 1.0
    return torch.stack((x_norm, y_norm, z_norm), dim=-1)


def sample_trilinear_parameter_map_points(
    parameter_map: torch.Tensor,
    coords_bev_xyz: torch.Tensor,
) -> torch.Tensor:
    if parameter_map.ndim != 4 or parameter_map.shape[0] != 1:
        raise ValueError(f"parameter_map must be [1,Z,Y,X], got {tuple(parameter_map.shape)}")
    grid = bev_xyz_to_grid_sample(coords_bev_xyz, tuple(int(v) for v in parameter_map.shape[1:]))
    volume = parameter_map[:, None, :, :, :]
    sampled = F.grid_sample(
        volume,
        grid.reshape(1, -1, 1, 1, 3),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=bool(RESAMPLER_ALIGN_CORNERS),
    )
    return sampled.reshape(-1)


def sample_trilinear_parameter_map_chunk(
    parameter_map: torch.Tensor,
    z0: int,
    z1: int,
    raw_y: int,
    raw_x: int,
    affine: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parameter_map.ndim != 4 or parameter_map.shape[0] != 1:
        raise ValueError(f"parameter_map must be [1,Z,Y,X], got {tuple(parameter_map.shape)}")
    parameter_shape = tuple(int(v) for v in parameter_map.shape[1:])
    coords_bev_xyz = SPLINE.raw_chunk_bev_coords_from_affine(
        z0,
        z1,
        raw_y,
        raw_x,
        affine,
        yy,
        xx,
        parameter_map.dtype,
        parameter_map.device,
    )
    valid = valid_bev_overlap_mask(coords_bev_xyz, parameter_shape).reshape(z1 - z0, raw_y, raw_x)
    grid = bev_xyz_to_grid_sample(coords_bev_xyz, parameter_shape).reshape(1, z1 - z0, raw_y, raw_x, 3)
    sampled = F.grid_sample(
        parameter_map[:, None, :, :, :],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=bool(RESAMPLER_ALIGN_CORNERS),
    )[0, 0]
    return sampled, valid


def sample_trilinear_bev_maps_chunk(
    bev_maps: list[torch.Tensor],
    z0: int,
    z1: int,
    raw_y: int,
    raw_x: int,
    affine: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bev_maps:
        raise ValueError("bev_maps must not be empty.")
    first_shape = tuple(int(v) for v in bev_maps[0].shape)
    if len(first_shape) != 4 or first_shape[0] != 1:
        raise ValueError(f"Each BEV map must be [1,Z,Y,X], got {first_shape}")
    for item in bev_maps:
        if tuple(int(v) for v in item.shape) != first_shape:
            raise ValueError(f"All BEV maps must have the same shape; got {tuple(item.shape)} vs {first_shape}")
    parameter_shape = tuple(int(v) for v in bev_maps[0].shape[1:])
    coords_bev_xyz = SPLINE.raw_chunk_bev_coords_from_affine(
        z0,
        z1,
        raw_y,
        raw_x,
        affine,
        yy,
        xx,
        bev_maps[0].dtype,
        bev_maps[0].device,
    )
    valid = valid_bev_overlap_mask(coords_bev_xyz, parameter_shape).reshape(z1 - z0, raw_y, raw_x)
    grid = bev_xyz_to_grid_sample(coords_bev_xyz, parameter_shape).reshape(1, z1 - z0, raw_y, raw_x, 3)
    volume = torch.cat([item[:, None, :, :, :] for item in bev_maps], dim=1)
    sampled = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=bool(RESAMPLER_ALIGN_CORNERS),
    )[0]
    return sampled, valid


def sample_trilinear_parameter_map_volume(
    parameter_map: torch.Tensor,
    raw_index_to_bev_index_affine: torch.Tensor,
    raw_shape_zyx: tuple[int, int, int],
    chunk_z: int = 16,
) -> torch.Tensor:
    if parameter_map.ndim != 4 or parameter_map.shape[0] != 1:
        raise ValueError(f"parameter_map must be [1,Z,Y,X], got {tuple(parameter_map.shape)}")
    if chunk_z <= 0:
        raise ValueError("chunk_z must be positive.")
    raw_z, raw_y, raw_x = (int(v) for v in raw_shape_zyx)
    y = torch.arange(raw_y, dtype=parameter_map.dtype, device=parameter_map.device)
    x = torch.arange(raw_x, dtype=parameter_map.dtype, device=parameter_map.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    chunks: list[torch.Tensor] = []
    for z0 in range(0, raw_z, chunk_z):
        z1 = min(z0 + chunk_z, raw_z)
        sampled, _ = sample_trilinear_parameter_map_chunk(
            parameter_map,
            z0,
            z1,
            raw_y,
            raw_x,
            raw_index_to_bev_index_affine,
            yy,
            xx,
        )
        chunks.append(sampled)
    return torch.cat(chunks, dim=0)


def sample_trilinear_bev_maps_volume(
    bev_maps: list[torch.Tensor],
    raw_index_to_bev_index_affine: torch.Tensor,
    raw_shape_zyx: tuple[int, int, int],
    chunk_z: int = 16,
) -> list[torch.Tensor]:
    if not bev_maps:
        raise ValueError("bev_maps must not be empty.")
    if chunk_z <= 0:
        raise ValueError("chunk_z must be positive.")
    raw_z, raw_y, raw_x = (int(v) for v in raw_shape_zyx)
    y = torch.arange(raw_y, dtype=bev_maps[0].dtype, device=bev_maps[0].device)
    x = torch.arange(raw_x, dtype=bev_maps[0].dtype, device=bev_maps[0].device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    chunk_lists: list[list[torch.Tensor]] = [[] for _ in bev_maps]
    for z0 in range(0, raw_z, chunk_z):
        z1 = min(z0 + chunk_z, raw_z)
        sampled, _ = sample_trilinear_bev_maps_chunk(
            bev_maps,
            z0,
            z1,
            raw_y,
            raw_x,
            raw_index_to_bev_index_affine,
            yy,
            xx,
        )
        for map_idx in range(len(bev_maps)):
            chunk_lists[map_idx].append(sampled[map_idx])
    return [torch.cat(chunks, dim=0) for chunks in chunk_lists]


def sample_raw_volume_to_bev_grid(
    raw_volume: torch.Tensor,
    raw_index_to_bev_index_affine: torch.Tensor,
    bev_shape_zyx: tuple[int, int, int],
    chunk_z: int = 16,
) -> torch.Tensor:
    if raw_volume.ndim != 3:
        raise ValueError(f"raw_volume must be [Z,Y,X], got {tuple(raw_volume.shape)}")
    if chunk_z <= 0:
        raise ValueError("chunk_z must be positive.")
    depth, height, width = (int(v) for v in bev_shape_zyx)
    raw_z, raw_y, raw_x = (int(v) for v in raw_volume.shape)
    affine = raw_index_to_bev_index_affine.to(device=raw_volume.device, dtype=raw_volume.dtype)
    if affine.shape != (4, 4):
        raise ValueError(f"raw_index_to_bev_index_affine must be [4,4], got {tuple(affine.shape)}")
    bev_to_raw = torch.linalg.inv(affine)
    y = torch.arange(height, dtype=raw_volume.dtype, device=raw_volume.device)
    x = torch.arange(width, dtype=raw_volume.dtype, device=raw_volume.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    raw_input = raw_volume[None, None, :, :, :]
    chunks: list[torch.Tensor] = []
    for z0 in range(0, depth, chunk_z):
        z1 = min(z0 + chunk_z, depth)
        z = torch.arange(z0, z1, dtype=raw_volume.dtype, device=raw_volume.device)[:, None, None]
        xx_chunk = xx[None, :, :].expand(z1 - z0, height, width)
        yy_chunk = yy[None, :, :].expand(z1 - z0, height, width)
        zz_chunk = z.expand(z1 - z0, height, width)
        ones = torch.ones_like(xx_chunk)
        bev_hom = torch.stack((xx_chunk, yy_chunk, zz_chunk, ones), dim=-1)
        raw_xyz = bev_hom @ bev_to_raw.T
        raw_x_coord = raw_xyz[..., 0]
        raw_y_coord = raw_xyz[..., 1]
        raw_z_coord = raw_xyz[..., 2]
        x_norm = raw_x_coord * (2.0 / float(max(raw_x - 1, 1))) - 1.0
        y_norm = raw_y_coord * (2.0 / float(max(raw_y - 1, 1))) - 1.0
        z_norm = raw_z_coord * (2.0 / float(max(raw_z - 1, 1))) - 1.0
        grid = torch.stack((x_norm, y_norm, z_norm), dim=-1)[None, :, :, :, :]
        sampled = F.grid_sample(
            raw_input,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=bool(RESAMPLER_ALIGN_CORNERS),
        )[0, 0]
        chunks.append(sampled)
    return torch.cat(chunks, dim=0)[None, :, :, :]


def idd_body_ct_threshold_normalized() -> float:
    return float(IDD_BODY_CT_THRESHOLD_HU) * float(CT_SCALE)


def fill_binary_holes_2d_stack_gpu(mask_stack: torch.Tensor) -> torch.Tensor:
    """Fill enclosed holes independently in a stack of 2D masks on GPU."""
    if mask_stack.ndim != 3:
        raise ValueError(f"mask_stack must be [N,H,W], got {tuple(mask_stack.shape)}")
    if not IDD_FILL_BODY_HOLES_2D:
        return mask_stack.to(dtype=torch.bool)
    mask = mask_stack.to(dtype=torch.bool)
    n_planes, height, width = (int(v) for v in mask.shape)
    if n_planes == 0 or height == 0 or width == 0:
        return mask
    mode = str(IDD_BODY_FILL_MODE).lower().strip()
    if mode == "axis_span":
        mask_i = mask.to(dtype=torch.int16)
        left_seen = torch.cumsum(mask_i, dim=2) > 0
        right_seen = torch.flip(torch.cumsum(torch.flip(mask_i, dims=(2,)), dim=2), dims=(2,)) > 0
        top_seen = torch.cumsum(mask_i, dim=1) > 0
        bottom_seen = torch.flip(torch.cumsum(torch.flip(mask_i, dims=(1,)), dim=1), dims=(1,)) > 0
        return mask | (left_seen & right_seen & top_seen & bottom_seen)
    if mode != "iterative_flood":
        raise ValueError(f"IDD_BODY_FILL_MODE must be 'axis_span' or 'iterative_flood', got {IDD_BODY_FILL_MODE!r}")

    background = ~mask
    exterior = torch.zeros_like(background)
    exterior[:, 0, :] = background[:, 0, :]
    exterior[:, -1, :] = background[:, -1, :]
    exterior[:, :, 0] = background[:, :, 0]
    exterior[:, :, -1] = background[:, :, -1]
    exterior_f = exterior.to(dtype=torch.float32)[:, None, :, :]
    background_f = background.to(dtype=torch.float32)[:, None, :, :]
    max_iterations = IDD_BODY_FILL_MAX_ITERATIONS
    if max_iterations is None:
        max_iterations = height + width
    if int(IDD_BODY_FILL_CONNECTIVITY) == 4:
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=exterior_f.dtype,
            device=exterior_f.device,
        ).reshape(1, 1, 3, 3)
        for _ in range(int(max_iterations)):
            exterior_f = (F.conv2d(exterior_f, kernel, padding=1) > 0.0).to(dtype=exterior_f.dtype) * background_f
    elif int(IDD_BODY_FILL_CONNECTIVITY) == 8:
        for _ in range(int(max_iterations)):
            exterior_f = F.max_pool2d(exterior_f, kernel_size=3, stride=1, padding=1) * background_f
    else:
        raise ValueError(f"IDD_BODY_FILL_CONNECTIVITY must be 4 or 8, got {IDD_BODY_FILL_CONNECTIVITY}")
    exterior = exterior_f[:, 0] > 0.5
    return mask | (background & ~exterior)


def bev_body_mask_from_ct(ct_bev: torch.Tensor) -> torch.Tensor:
    if ct_bev.ndim != 4 or ct_bev.shape[0] != 1:
        raise ValueError(f"ct_bev must be [1,Z,Y,X], got {tuple(ct_bev.shape)}")
    threshold = idd_body_ct_threshold_normalized()
    mask_zyx = ct_bev[0] > threshold
    stack_y_zx = mask_zyx.permute(1, 0, 2).contiguous()
    filled_y_zx = fill_binary_holes_2d_stack_gpu(stack_y_zx)
    return filled_y_zx.permute(1, 0, 2).contiguous()


def raw_body_mask_from_sampled_ct(ct_raw: torch.Tensor) -> torch.Tensor:
    if ct_raw.ndim != 3:
        raise ValueError(f"ct_raw must be [Z,Y,X], got {tuple(ct_raw.shape)}")
    threshold = idd_body_ct_threshold_normalized()
    mask_zyx = ct_raw > threshold
    return fill_binary_holes_2d_stack_gpu(mask_zyx)


def _as_batched_bev_volume(volume: torch.Tensor, name: str) -> torch.Tensor:
    if volume.ndim == 3:
        return volume[None, :, :, :]
    if volume.ndim == 4:
        return volume
    raise ValueError(f"{name} must be [Z,Y,X] or [B,Z,Y,X], got {tuple(volume.shape)}")


def compute_bev_idd_rmse_loss(
    prediction_bev: torch.Tensor,
    reference_bev: torch.Tensor,
    body_mask_bev: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = _as_batched_bev_volume(prediction_bev, "prediction_bev")
    ref = _as_batched_bev_volume(reference_bev, "reference_bev").to(device=pred.device, dtype=pred.dtype)
    if pred.shape != ref.shape:
        raise ValueError(f"IDD BEV shape mismatch: {tuple(pred.shape)} vs {tuple(ref.shape)}")
    if body_mask_bev is not None:
        mask = _as_batched_bev_volume(body_mask_bev, "body_mask_bev").to(device=pred.device, dtype=torch.bool)
        if mask.shape != pred.shape:
            raise ValueError(f"IDD body mask shape mismatch: {tuple(mask.shape)} vs {tuple(pred.shape)}")
        mask_f = mask.to(dtype=pred.dtype)
        pred = pred * mask_f
        ref = ref * mask_f
    pred_idd = torch.sum(pred, dim=(2, 3))
    ref_idd = torch.sum(ref, dim=(2, 3))
    scale = torch.clamp(torch.amax(torch.abs(ref_idd), dim=1, keepdim=True), min=float(IDD_LOSS_EPS))
    diff = (pred_idd - ref_idd) / scale
    return torch.sqrt(torch.mean(diff * diff, dim=1) + float(IDD_LOSS_EPS)).mean()


def compute_final_roundtrip_idd_loss(
    bev_map: torch.Tensor,
    residual_map: torch.Tensor,
    ct_bev: torch.Tensor,
    dose_raw: torch.Tensor,
    affine: torch.Tensor,
) -> torch.Tensor:
    if bev_map.ndim != 4 or bev_map.shape[0] != 1:
        raise ValueError(f"bev_map must be [1,Z,Y,X], got {tuple(bev_map.shape)}")
    if residual_map.ndim != 4 or residual_map.shape != bev_map.shape:
        raise ValueError(f"residual_map must match bev_map, got {tuple(residual_map.shape)} vs {tuple(bev_map.shape)}")
    if ct_bev.ndim != 4 or ct_bev.shape != bev_map.shape:
        raise ValueError(f"ct_bev must match bev_map, got {tuple(ct_bev.shape)} vs {tuple(bev_map.shape)}")
    if dose_raw.ndim != 3:
        raise ValueError(f"dose_raw must be [Z,Y,X], got {tuple(dose_raw.shape)}")
    raw_shape_zyx = tuple(int(v) for v in dose_raw.shape)
    final_bev = bev_map + residual_map
    final_raw, ct_raw = sample_trilinear_bev_maps_volume(
        [final_bev, ct_bev],
        affine,
        raw_shape_zyx,
        chunk_z=RAW_FULL_OVERLAP_CHUNK_Z,
    )
    raw_body_mask = raw_body_mask_from_sampled_ct(ct_raw)
    raw_body_mask_f = raw_body_mask.to(dtype=final_raw.dtype)
    final_raw = torch.clamp(final_raw, min=0.0) * raw_body_mask_f
    dose_raw_masked = dose_raw.to(device=final_raw.device, dtype=final_raw.dtype) * raw_body_mask_f
    final_bev_roundtrip = sample_raw_volume_to_bev_grid(
        final_raw,
        affine,
        tuple(int(v) for v in final_bev.shape[1:]),
        chunk_z=IDD_PATIENT_TO_BEV_CHUNK_Z,
    )
    reference_bev_roundtrip = sample_raw_volume_to_bev_grid(
        dose_raw_masked,
        affine,
        tuple(int(v) for v in final_bev.shape[1:]),
        chunk_z=IDD_PATIENT_TO_BEV_CHUNK_Z,
    )
    return compute_bev_idd_rmse_loss(final_bev_roundtrip, reference_bev_roundtrip)


def compute_patient_grid_mlc_ct_region_losses_and_metrics(
    parameter_map: torch.Tensor,
    ct_bev: torch.Tensor,
    mlc_bev: torch.Tensor,
    dose_raw: torch.Tensor,
    affine: torch.Tensor,
    forward_time_s: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if parameter_map.ndim != 4 or parameter_map.shape[0] != 1:
        raise ValueError(f"parameter_map must be [1,Z,Y,X], got {tuple(parameter_map.shape)}")
    if ct_bev.ndim != 4 or ct_bev.shape[0] != 1:
        raise ValueError(f"ct_bev must be [1,Z,Y,X], got {tuple(ct_bev.shape)}")
    if mlc_bev.ndim != 4 or mlc_bev.shape[0] != 1:
        raise ValueError(f"mlc_bev must be [1,Z,Y,X], got {tuple(mlc_bev.shape)}")
    if dose_raw.ndim != 3:
        raise ValueError(f"dose_raw must be [Z,Y,X], got {tuple(dose_raw.shape)}")

    chunk_z = int(RAW_FULL_OVERLAP_CHUNK_Z)
    if chunk_z <= 0:
        raise ValueError("RAW_FULL_OVERLAP_CHUNK_Z must be positive.")

    raw_z, raw_y, raw_x = (int(v) for v in dose_raw.shape)
    ref_max = torch.clamp(torch.max(dose_raw), min=1e-12)
    y = torch.arange(raw_y, dtype=parameter_map.dtype, device=parameter_map.device)
    x = torch.arange(raw_x, dtype=parameter_map.dtype, device=parameter_map.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    zero = parameter_map.sum() * 0.0
    global_abs_sum = zero
    mlc_full_abs_sum = zero
    mlc_penumbra_abs_sum = zero
    outside_low_abs_sum = zero
    dose_region_abs_sum = zero
    dose_region_10_30_abs_sum = zero
    dose_region_30_80_abs_sum = zero
    dose_region_80_100_abs_sum = zero
    dose_region_squared_sum = zero
    squared_sum = zero
    negative_point_sum = zero

    global_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    mlc_full_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    mlc_penumbra_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    outside_low_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    dose_region_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    body_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    pred_min = torch.full((), float("inf"), dtype=parameter_map.dtype, device=parameter_map.device)
    pred_max = torch.full((), float("-inf"), dtype=parameter_map.dtype, device=parameter_map.device)

    for z0 in range(0, raw_z, chunk_z):
        z1 = min(z0 + chunk_z, raw_z)
        sampled, valid = sample_trilinear_bev_maps_chunk(
            [parameter_map, ct_bev, mlc_bev],
            z0,
            z1,
            raw_y,
            raw_x,
            affine,
            yy,
            xx,
        )
        pred_raw_points = sampled[0].reshape(-1)
        ct_raw_points = sampled[1].reshape(-1)
        mlc_raw_points = sampled[2].reshape(-1)
        valid_flat = valid.reshape(-1)
        label_raw_points = dose_raw[z0:z1].reshape(-1).to(dtype=pred_raw_points.dtype)

        if RAW_FULL_OVERLAP_CHECK_FINITE_PER_CHUNK:
            if not torch.isfinite(pred_raw_points).all() or not torch.isfinite(label_raw_points).all():
                raise RuntimeError("NaN/Inf encountered in patient-grid predictions or labels.")

        diff = pred_raw_points - label_raw_points
        abs_diff = torch.abs(diff)
        valid_f = valid_flat.to(dtype=parameter_map.dtype)
        body = valid_flat & (ct_raw_points > float(PATIENT_GRID_CT_BODY_THRESHOLD))
        mlc_full = body & (mlc_raw_points >= 1.0 - float(PATIENT_GRID_MLC_FULL_EPS))
        mlc_penumbra = (
            body
            & (mlc_raw_points > float(PATIENT_GRID_MLC_ZERO_EPS))
            & (mlc_raw_points < 1.0 - float(PATIENT_GRID_MLC_FULL_EPS))
        )
        outside_low = (
            body
            & (mlc_raw_points <= float(PATIENT_GRID_MLC_ZERO_EPS))
            & (label_raw_points > float(PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION) * ref_max)
        )
        dose_region = valid_flat & (label_raw_points >= float(DOSE_REGION_THRESHOLD) * ref_max)

        mlc_full_f = mlc_full.to(dtype=parameter_map.dtype)
        mlc_penumbra_f = mlc_penumbra.to(dtype=parameter_map.dtype)
        outside_low_f = outside_low.to(dtype=parameter_map.dtype)
        dose_region_f = dose_region.to(dtype=parameter_map.dtype)
        body_f = body.to(dtype=parameter_map.dtype)

        global_abs_sum = global_abs_sum + torch.sum(abs_diff * valid_f)
        squared_sum = squared_sum + torch.sum(diff * diff * valid_f)
        dose_region_abs_sum = dose_region_abs_sum + torch.sum(abs_diff * dose_region_f)
        dose_region_squared_sum = dose_region_squared_sum + torch.sum(diff * diff * dose_region_f)
        global_count = global_count + torch.sum(valid_f)
        dose_region_count = dose_region_count + torch.sum(dose_region_f)
        body_count = body_count + torch.sum(body_f)
        mlc_full_abs_sum = mlc_full_abs_sum + torch.sum(abs_diff * mlc_full_f)
        mlc_penumbra_abs_sum = mlc_penumbra_abs_sum + torch.sum(abs_diff * mlc_penumbra_f)
        outside_low_abs_sum = outside_low_abs_sum + torch.sum(abs_diff * outside_low_f)
        mlc_full_count = mlc_full_count + torch.sum(mlc_full_f)
        mlc_penumbra_count = mlc_penumbra_count + torch.sum(mlc_penumbra_f)
        outside_low_count = outside_low_count + torch.sum(outside_low_f)
        negative_point_sum = negative_point_sum + torch.sum(((pred_raw_points < 0) & valid_flat).to(dtype=parameter_map.dtype))
        pred_min = torch.minimum(pred_min, torch.min(torch.where(valid_flat, pred_raw_points.detach(), torch.full_like(pred_raw_points, float("inf")))))
        pred_max = torch.maximum(pred_max, torch.max(torch.where(valid_flat, pred_raw_points.detach(), torch.full_like(pred_raw_points, float("-inf")))))

    if int(global_count.detach().cpu()) == 0:
        raise RuntimeError("No raw dose voxel centers overlap the BEV output grid.")

    global_mae = global_abs_sum / torch.clamp(global_count, min=1.0)
    mlc_full_mae = mlc_full_abs_sum / torch.clamp(mlc_full_count, min=1.0)
    mlc_penumbra_mae = mlc_penumbra_abs_sum / torch.clamp(mlc_penumbra_count, min=1.0)
    outside_low_mae = outside_low_abs_sum / torch.clamp(outside_low_count, min=1.0)
    global_mse = squared_sum / torch.clamp(global_count, min=1.0)
    dose_region_mae = dose_region_abs_sum / torch.clamp(dose_region_count, min=1.0)
    dose_region_mse = dose_region_squared_sum / torch.clamp(dose_region_count, min=1.0)
    total_loss = (
        float(PATIENT_GRID_GLOBAL_MAE_WEIGHT) * global_mae
        + float(PATIENT_GRID_MLC_FULL_MAE_WEIGHT) * mlc_full_mae
        + float(PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT) * mlc_penumbra_mae
        + float(PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT) * outside_low_mae
    )

    with torch.no_grad():
        param = parameter_map.detach()
        metrics = {
            "patient_grid_global_mae": float(global_mae.detach().cpu()),
            "patient_grid_mlc_full_mae": float(mlc_full_mae.detach().cpu()),
            "patient_grid_mlc_penumbra_mae": float(mlc_penumbra_mae.detach().cpu()),
            "patient_grid_outside_low_dose_mae": float(outside_low_mae.detach().cpu()),
            "patient_grid_region_loss": float(total_loss.detach().cpu()),
            "global_mse": float(global_mse.detach().cpu()),
            "global_mae": float(global_mae.detach().cpu()),
            "raw_point_mse": float(global_mse.detach().cpu()),
            "raw_point_mae": float(global_mae.detach().cpu()),
            "dose_region_mse": float(dose_region_mse.detach().cpu()),
            "dose_region_mae": float(dose_region_mae.detach().cpu()),
            "raw_high_region_mae": float(dose_region_mae.detach().cpu()),
            "raw_mid_region_mae": float("nan"),
            "negative_prediction_fraction": float((negative_point_sum / torch.clamp(global_count, min=1.0)).detach().cpu()),
            "patient_grid_final_lowest_value": float(pred_min.detach().cpu()),
            "patient_grid_max_prediction_over_max_ref": float((pred_max / ref_max).detach().cpu()),
            "sampled_prediction_min": float(pred_min.detach().cpu()),
            "sampled_prediction_max": float(pred_max.detach().cpu()),
            "prediction_reference_max_ratio": float((pred_max / ref_max).detach().cpu()),
            "coefficient_min": float(torch.min(param).cpu()),
            "coefficient_max": float(torch.max(param).cpu()),
            "coefficient_mean": float(torch.mean(param).cpu()),
            "coefficient_std": float(torch.std(param).cpu()),
            "parameter_min": float(torch.min(param).cpu()),
            "parameter_max": float(torch.max(param).cpu()),
            "parameter_mean": float(torch.mean(param).cpu()),
            "parameter_std": float(torch.std(param).cpu()),
            "raw_loss": float(total_loss.detach().cpu()),
            "global_loss": float(global_mae.detach().cpu()),
            "dose_region_loss": float(dose_region_mae.detach().cpu()),
            "negative_loss": 0.0,
            "bev_aux_loss": 0.0,
            "bev_grid_anchor_loss": 0.0,
            "model_forward_time_s": float(forward_time_s),
        }
    return total_loss, metrics


def compute_full_overlap_losses_and_metrics(
    parameter_map: torch.Tensor,
    dose_raw: torch.Tensor,
    dose_bev: torch.Tensor,
    affine: torch.Tensor,
    spline_time_s: float,
    forward_time_s: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if parameter_map.ndim != 4 or parameter_map.shape[0] != 1:
        raise ValueError(f"Full-overlap loss expects parameter_map [1,Z,Y,X], got {tuple(parameter_map.shape)}")
    if dose_raw.ndim != 3:
        raise ValueError(f"dose_raw must be [Z,Y,X], got {tuple(dose_raw.shape)}")
    chunk_z = int(RAW_FULL_OVERLAP_CHUNK_Z)
    if chunk_z <= 0:
        raise ValueError("RAW_FULL_OVERLAP_CHUNK_Z must be positive.")

    raw_z, raw_y, raw_x = (int(v) for v in dose_raw.shape)
    ref_max = torch.clamp(torch.max(dose_raw), min=1e-12)
    y = torch.arange(raw_y, dtype=parameter_map.dtype, device=parameter_map.device)
    x = torch.arange(raw_x, dtype=parameter_map.dtype, device=parameter_map.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    zero = parameter_map.sum() * 0.0
    global_sum = zero
    dose_region_sum = zero
    squared_sum = zero
    dose_region_squared_sum = zero
    negative_sum = zero
    negative_point_sum = zero
    mid_abs_sum = zero
    global_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    dose_region_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    mid_count = torch.zeros((), dtype=parameter_map.dtype, device=parameter_map.device)
    pred_min = torch.full((), float("inf"), dtype=parameter_map.dtype, device=parameter_map.device)
    pred_max = torch.full((), float("-inf"), dtype=parameter_map.dtype, device=parameter_map.device)

    for z0 in range(0, raw_z, chunk_z):
        z1 = min(z0 + chunk_z, raw_z)
        pred_chunk, valid = sample_trilinear_parameter_map_chunk(
            parameter_map,
            z0,
            z1,
            raw_y,
            raw_x,
            affine,
            yy,
            xx,
        )
        valid_count = valid.sum()

        pred_raw_points = pred_chunk.reshape(-1)
        valid_flat = valid.reshape(-1)
        label_raw_points = dose_raw[z0:z1].reshape(-1).to(dtype=pred_raw_points.dtype)
        if RAW_FULL_OVERLAP_CHECK_FINITE_PER_CHUNK:
            if not torch.isfinite(pred_raw_points).all() or not torch.isfinite(label_raw_points).all():
                raise RuntimeError("NaN/Inf encountered in full-overlap predictions or labels.")

        diff = pred_raw_points - label_raw_points
        abs_diff = torch.abs(diff)
        valid_f = valid_flat.to(dtype=parameter_map.dtype)
        dose_region = valid_flat & (label_raw_points >= DOSE_REGION_THRESHOLD * ref_max)
        dose_region_f = dose_region.to(dtype=parameter_map.dtype)
        mid = valid_flat & (label_raw_points >= 0.01 * ref_max) & (label_raw_points < DOSE_REGION_THRESHOLD * ref_max)
        mid_f = mid.to(dtype=parameter_map.dtype)

        global_sum = global_sum + torch.sum(pointwise_loss(diff, RAW_GLOBAL_LOSS_TYPE) * valid_f)
        squared_sum = squared_sum + torch.sum(diff * diff * valid_f)
        global_count = global_count + valid_count.to(dtype=global_count.dtype)
        negative_values = torch.relu(-pred_raw_points)
        negative_sum = negative_sum + torch.sum(negative_values * negative_values * valid_f)
        negative_point_sum = negative_point_sum + torch.sum(((pred_raw_points < 0) & valid_flat).to(dtype=parameter_map.dtype))
        pred_min = torch.minimum(pred_min, torch.min(torch.where(valid_flat, pred_raw_points.detach(), torch.full_like(pred_raw_points, float("inf")))))
        pred_max = torch.maximum(pred_max, torch.max(torch.where(valid_flat, pred_raw_points.detach(), torch.full_like(pred_raw_points, float("-inf")))))

        dose_region_sum = dose_region_sum + torch.sum(pointwise_loss(diff, RAW_DOSE_REGION_LOSS_TYPE) * dose_region_f)
        dose_region_squared_sum = dose_region_squared_sum + torch.sum(diff * diff * dose_region_f)
        dose_region_count = dose_region_count + torch.sum(dose_region_f)
        mid_abs_sum = mid_abs_sum + torch.sum(abs_diff * mid_f)
        mid_count = mid_count + torch.sum(mid_f)

    if int(global_count.detach().cpu()) == 0:
        raise RuntimeError("No raw dose voxel centers overlap the BEV output grid.")

    global_loss = global_sum / torch.clamp(global_count, min=1.0)
    dose_region_loss = dose_region_sum / torch.clamp(dose_region_count, min=1.0)
    raw_loss = RAW_GLOBAL_LOSS_WEIGHT * global_loss + RAW_DOSE_REGION_LOSS_WEIGHT * dose_region_loss
    negative_loss = negative_sum / torch.clamp(global_count, min=1.0)
    bev_aux_loss = parameter_map.sum() * 0.0
    if BEV_AUX_LOSS_WEIGHT != 0.0:
        bev_aux_loss = reduced_loss(parameter_map[0], dose_bev, RAW_GLOBAL_LOSS_TYPE)
    bev_grid_anchor_loss, bev_grid_metrics = compute_bev_grid_anchor_loss_and_metrics(parameter_map, dose_bev)
    total_loss = (
        raw_loss
        + bev_grid_anchor_loss
        + NEGATIVE_DOSE_LOSS_WEIGHT * negative_loss
        + BEV_AUX_LOSS_WEIGHT * bev_aux_loss
    )

    with torch.no_grad():
        param = parameter_map.detach()
        global_mse = squared_sum / torch.clamp(global_count, min=1.0)
        global_mae = global_sum / torch.clamp(global_count, min=1.0) if validate_loss_type(RAW_GLOBAL_LOSS_TYPE) == "l1" else torch.full_like(global_loss, float("nan"))
        dose_region_mse = dose_region_squared_sum / torch.clamp(dose_region_count, min=1.0)
        dose_region_mae = dose_region_sum / torch.clamp(dose_region_count, min=1.0) if validate_loss_type(RAW_DOSE_REGION_LOSS_TYPE) == "l1" else torch.full_like(dose_region_loss, float("nan"))
        dose_region_point_fraction = dose_region_count / torch.clamp(global_count, min=1.0)
        mid_region_mae = mid_abs_sum / torch.clamp(mid_count, min=1.0)
        has_dose_region = int(dose_region_count.detach().cpu()) > 0
        has_mid_region = int(mid_count.detach().cpu()) > 0
        metrics = {
            "global_mse": float(global_mse.detach().cpu()),
            "global_mae": float(global_mae.detach().cpu()),
            "dose_region_mse": float(dose_region_mse.detach().cpu()) if has_dose_region else float("nan"),
            "dose_region_mae": float(dose_region_mae.detach().cpu()) if has_dose_region else float("nan"),
            "dose_region_point_fraction": float(dose_region_point_fraction.detach().cpu()),
            "raw_point_mse": float(global_mse.detach().cpu()),
            "raw_point_mae": float(global_mae.detach().cpu()),
            "raw_high_region_mae": float(dose_region_mae.detach().cpu()) if has_dose_region else float("nan"),
            "raw_mid_region_mae": float(mid_region_mae.detach().cpu()) if has_mid_region else float("nan"),
            "raw_overlap_fraction": float((global_count / float(raw_z * raw_y * raw_x)).detach().cpu()),
            "negative_prediction_fraction": float((negative_point_sum / torch.clamp(global_count, min=1.0)).detach().cpu()),
            "sampled_prediction_min": float(pred_min.detach().cpu()),
            "sampled_prediction_max": float(pred_max.detach().cpu()),
            "prediction_reference_max_ratio": float((pred_max / ref_max).detach().cpu()),
            "coefficient_min": float(torch.min(param).cpu()),
            "coefficient_max": float(torch.max(param).cpu()),
            "coefficient_mean": float(torch.mean(param).cpu()),
            "coefficient_std": float(torch.std(param).cpu()),
            "parameter_min": float(torch.min(param).cpu()),
            "parameter_max": float(torch.max(param).cpu()),
            "parameter_mean": float(torch.mean(param).cpu()),
            "parameter_std": float(torch.std(param).cpu()),
            "global_loss": float(global_loss.detach().cpu()),
            "dose_region_loss": float(dose_region_loss.detach().cpu()) if has_dose_region else float("nan"),
            "raw_loss": float(raw_loss.detach().cpu()),
            "negative_loss": float(negative_loss.detach().cpu()),
            "bev_aux_loss": float(bev_aux_loss.detach().cpu()),
            **bev_grid_metrics,
            "spline_sampling_time_s": float(spline_time_s),
            "model_forward_time_s": float(forward_time_s),
        }
    return total_loss, metrics


def add_scalars(writer, prefix: str, metrics: dict[str, float], step: int) -> None:
    for key, value in metrics.items():
        if np.isfinite(value):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def _write_scalar_if_finite(writer, tag: str, value: float, step: int) -> None:
    if np.isfinite(value):
        writer.add_scalar(tag, value, step)


def add_curated_training_scalars(writer, split: str, loss_value: float, metrics: dict[str, float], step: int) -> None:
    _write_scalar_if_finite(writer, f"{split}/loss", loss_value, step)

    bev_loss_keys = {
        "weighted_mae": "bev_branch_weighted_mae",
        "gradient_weighted_mae": "bev_branch_gradient_weighted_mae",
        "global_mae": "bev_branch_global_mae",
        "mlc_full_region_mae": "bev_branch_mlc_full_mae",
        "penumbra_mae": "bev_branch_mlc_penumbra_mae",
        "low_dose_mae": "bev_branch_outside_low_dose_mae",
        "total": "bev_branch_loss",
    }
    patient_loss_keys = {
        "weighted_mae": "patient_grid_weighted_mae",
        "gradient_weighted_mae": "patient_grid_gradient_weighted_mae",
        "global_mae": "patient_grid_global_mae",
        "mlc_full_region_mae": "patient_grid_mlc_full_mae",
        "penumbra_mae": "patient_grid_mlc_penumbra_mae",
        "low_dose_mae": "patient_grid_outside_low_dose_mae",
        "dose_region_mae": "dose_region_mae",
        "total": "patient_branch_loss",
    }
    for out_name, metric_name in bev_loss_keys.items():
        _write_scalar_if_finite(writer, f"{split}_losses_bev/{out_name}", float(metrics.get(metric_name, float("nan"))), step)
    for out_name, metric_name in patient_loss_keys.items():
        _write_scalar_if_finite(writer, f"{split}_losses_patient/{out_name}", float(metrics.get(metric_name, float("nan"))), step)

    _write_scalar_if_finite(writer, f"{split}_losses_idd/bev_raw", float(metrics.get("idd_bev_loss", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_losses_idd/final_roundtrip", float(metrics.get("idd_final_loss", float("nan"))), step)

    _write_scalar_if_finite(writer, f"{split}_losses/total", loss_value, step)
    _write_scalar_if_finite(writer, f"{split}_monitor_bev/global_mae", float(metrics.get("bev_branch_global_mae", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_bev/dose_region_mae", float(metrics.get("bev_branch_dose_region_mae", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_bev/final_lowest_value", float(metrics.get("bev_branch_final_lowest_value", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_bev/max_prediction_over_max_ref", float(metrics.get("bev_branch_max_prediction_over_max_ref", float("nan"))), step)

    _write_scalar_if_finite(writer, f"{split}_monitor_patient/global_mae", float(metrics.get("patient_grid_global_mae", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_patient/dose_region_mae", float(metrics.get("dose_region_mae", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_patient/final_lowest_value", float(metrics.get("patient_grid_final_lowest_value", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_patient/max_prediction_over_max_ref", float(metrics.get("patient_grid_max_prediction_over_max_ref", float("nan"))), step)

    _write_scalar_if_finite(writer, f"{split}_negative_values/patient_fraction", float(metrics.get("patient_grid_negative_prediction_fraction", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_negative_values/patient_value_min", float(metrics.get("patient_grid_negative_value_min", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_negative_values/patient_value_p05", float(metrics.get("patient_grid_negative_value_p05", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_negative_values/patient_value_p95", float(metrics.get("patient_grid_negative_value_p95", float("nan"))), step)

    _write_scalar_if_finite(writer, f"{split}_monitor_residual/raw_mean_abs", float(metrics.get("residual_raw_mean_abs", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_residual/raw_min", float(metrics.get("residual_raw_min", float("nan"))), step)
    _write_scalar_if_finite(writer, f"{split}_monitor_residual/raw_max", float(metrics.get("residual_raw_max", float("nan"))), step)


def empty_metric_accumulator() -> dict[str, float]:
    return {}


def add_to_accumulator(accumulator: dict[str, float], metrics: dict[str, float], weight: float = 1.0) -> None:
    for key, value in metrics.items():
        if np.isfinite(value):
            accumulator[key] = accumulator.get(key, 0.0) + float(value) * float(weight)


def average_accumulator(accumulator: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in accumulator.items()}


def epoch_accumulator_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if "negative" not in key
    }


def batch_to_device(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    batch = ensure_batched(batch)
    inputs = tuple(tensor.to(device, non_blocking=True) for tensor in batch["inputs"])  # type: ignore[union-attr]
    aperture_features = batch["aperture_features"].to(device, non_blocking=True)  # type: ignore[union-attr]
    angle_features = batch["angle_features"].to(device, non_blocking=True)  # type: ignore[union-attr]
    dose_raw_list = [tensor.to(device, non_blocking=True) for tensor in batch["dose_raw"]]  # type: ignore[union-attr]
    dose_bev = batch["dose_bev"].to(device, non_blocking=True)  # type: ignore[union-attr]
    affine_list = [tensor.to(device, non_blocking=True) for tensor in batch["raw_index_to_bev"]]  # type: ignore[union-attr]
    return inputs, aperture_features, angle_features, dose_raw_list, dose_bev, affine_list


def _mean_abs_film_delta_from_embeddings(
    film_modules: tuple[nn.Module, ...],
    branch_head_name: str,
    original_embedding: torch.Tensor,
    ablated_embedding: torch.Tensor,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for film_module in film_modules:
        head = getattr(film_module, branch_head_name)
        original_delta = head(original_embedding)
        ablated_delta = head(ablated_embedding)
        values.append(torch.mean(torch.abs(original_delta - ablated_delta)))
    return torch.stack(values).mean()


def film_contribution_metrics(
    model,
    aperture_features: torch.Tensor,
    angle_features: torch.Tensor,
    sample_key: str,
) -> dict[str, float]:
    module = model.module if hasattr(model, "module") else model
    film_modules = (module.film64, module.film32, module.film16)
    with torch.no_grad():
        aperture_embedding = module.aperture_condition_encoder(aperture_features)
        angle_embedding = module.angle_condition_encoder(angle_features)

        area_ablated = aperture_features.clone()
        area_ablated[:, 0] = 0.0
        area_ablated_embedding = module.aperture_condition_encoder(area_ablated)

        s_ablated = aperture_features.clone()
        s_ablated[:, 1] = 0.0
        s_ablated_embedding = module.aperture_condition_encoder(s_ablated)

        angle_ablated_embedding = module.angle_condition_encoder(torch.zeros_like(angle_features))

        metrics_t = {
            "a_opening_mm2": _mean_abs_film_delta_from_embeddings(
                film_modules,
                "aperture_head",
                aperture_embedding,
                area_ablated_embedding,
            ),
            "s_equivalent_mm": _mean_abs_film_delta_from_embeddings(
                film_modules,
                "aperture_head",
                aperture_embedding,
                s_ablated_embedding,
            ),
            "gantry_angle": _mean_abs_film_delta_from_embeddings(
                film_modules,
                "angle_head",
                angle_embedding,
                angle_ablated_embedding,
            ),
        }
        out: dict[str, float] = {}
        for key, value in metrics_t.items():
            if not torch.isfinite(value):
                raise RuntimeError(f"NaN/Inf FiLM contribution {key} for sample {sample_key}")
            out[key] = float(value.detach().cpu())
        return out


def skip_contribution_metrics(model) -> dict[str, float]:
    module = model.module if hasattr(model, "module") else model
    stats = getattr(module, "last_skip_stats", {})
    out: dict[str, float] = {}
    with torch.no_grad():
        for key, value in stats.items():
            if not torch.is_tensor(value):
                continue
            if not torch.isfinite(value).all():
                raise RuntimeError(f"NaN/Inf skip contribution metric {key}")
            out[key] = float(value.detach().mean().cpu())
    return out


def add_contribution_scalars(writer, split: str, film_metrics: dict[str, float], skip_metrics: dict[str, float], step: int) -> None:
    for key in ("a_opening_mm2", "s_equivalent_mm", "gantry_angle"):
        _write_scalar_if_finite(writer, f"{split}_contributions/film_{key}", float(film_metrics.get(key, float("nan"))), step)
    for key, value in skip_metrics.items():
        _write_scalar_if_finite(writer, f"{split}_contributions/{key}", value, step)


def stable_sample_seed(sample_key: str, base_seed: int) -> int:
    return (int(base_seed) + zlib.crc32(sample_key.encode("utf-8"))) % (2**31 - 1)


def compute_bev_mlc_ct_region_losses_and_metrics(
    bev_map: torch.Tensor,
    ct_bev: torch.Tensor,
    mlc_bev: torch.Tensor,
    dose_bev: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if bev_map.ndim != 4 or bev_map.shape[0] != 1:
        raise ValueError(f"bev_map must be [1,Z,Y,X], got {tuple(bev_map.shape)}")
    if ct_bev.ndim != 4 or ct_bev.shape[0] != 1:
        raise ValueError(f"ct_bev must be [1,Z,Y,X], got {tuple(ct_bev.shape)}")
    if mlc_bev.ndim != 4 or mlc_bev.shape[0] != 1:
        raise ValueError(f"mlc_bev must be [1,Z,Y,X], got {tuple(mlc_bev.shape)}")
    if dose_bev.ndim == 4 and dose_bev.shape[0] == 1:
        dose_bev = dose_bev[0]
    if dose_bev.ndim != 3:
        raise ValueError(f"dose_bev must be [Z,Y,X], got {tuple(dose_bev.shape)}")

    pred = bev_map[0]
    ref = dose_bev.to(device=pred.device, dtype=pred.dtype)
    ct = ct_bev[0].to(device=pred.device, dtype=pred.dtype)
    mlc = mlc_bev[0].to(device=pred.device, dtype=pred.dtype)
    if pred.shape != ref.shape:
        raise ValueError(f"BEV prediction/reference shape mismatch: {tuple(pred.shape)} vs {tuple(ref.shape)}")

    ref_max = torch.clamp(torch.max(ref), min=1e-12)
    pred_for_mae = leaky_clamp_prediction_for_mae(pred)
    diff = pred_for_mae - ref
    abs_diff = torch.abs(diff)
    body = ct > float(PATIENT_GRID_CT_BODY_THRESHOLD)
    mlc_full = body & (mlc >= 1.0 - float(PATIENT_GRID_MLC_FULL_EPS))
    mlc_penumbra = (
        body
        & (mlc > float(PATIENT_GRID_MLC_ZERO_EPS))
        & (mlc < 1.0 - float(PATIENT_GRID_MLC_FULL_EPS))
    )
    outside_low = (
        body
        & (mlc <= float(PATIENT_GRID_MLC_ZERO_EPS))
        & (ref > float(PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION) * ref_max)
    )
    dose_region = ref >= float(DOSE_REGION_THRESHOLD) * ref_max

    global_mae = torch.mean(abs_diff)
    mlc_full_mae, mlc_full_count = masked_l1_mean_no_sync(diff, mlc_full)
    mlc_penumbra_mae, mlc_penumbra_count = masked_l1_mean_no_sync(diff, mlc_penumbra)
    outside_low_mae, outside_low_count = masked_l1_mean_no_sync(diff, outside_low)
    dose_region_mae, _ = masked_l1_mean_no_sync(diff, dose_region)
    total_loss = (
        float(PATIENT_GRID_GLOBAL_MAE_WEIGHT) * global_mae
        + float(PATIENT_GRID_MLC_FULL_MAE_WEIGHT) * mlc_full_mae
        + float(PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT) * mlc_penumbra_mae
        + float(PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT) * outside_low_mae
    )

    with torch.no_grad():
        metrics = {
            "bev_branch_loss": float(total_loss.detach().cpu()),
            "bev_branch_weighted_mae": float(total_loss.detach().cpu()),
            "bev_branch_mlc_full_mae": float(mlc_full_mae.detach().cpu()),
            "bev_branch_mlc_penumbra_mae": float(mlc_penumbra_mae.detach().cpu()),
            "bev_branch_outside_low_dose_mae": float(outside_low_mae.detach().cpu()),
            "bev_branch_dose_region_mae": float(dose_region_mae.detach().cpu()),
            "bev_branch_mlc_full_count": float(mlc_full_count.detach().cpu()),
            "bev_branch_mlc_penumbra_count": float(mlc_penumbra_count.detach().cpu()),
            "bev_branch_outside_low_dose_count": float(outside_low_count.detach().cpu()),
            "bev_branch_gradient_weighted_mae": float("nan"),
            "bev_branch_gradient_weight_sum": 0.0,
            "bev_branch_global_mae": float(global_mae.detach().cpu()),
            "bev_branch_final_lowest_value": float(torch.min(pred.detach()).cpu()),
            "bev_branch_prediction_min": float(torch.min(pred.detach()).cpu()),
            "bev_branch_prediction_max": float(torch.max(pred.detach()).cpu()),
            "bev_branch_max_prediction_over_max_ref": float((torch.max(pred.detach()) / ref_max).cpu()),
        }
    return total_loss, metrics


def compute_patient_grid_residual_losses_and_metrics(
    bev_map_detached: torch.Tensor,
    residual_map: torch.Tensor,
    ct_bev: torch.Tensor,
    mlc_bev: torch.Tensor,
    dose_raw: torch.Tensor,
    affine: torch.Tensor,
    forward_time_s: float,
    compute_negative_value_monitor: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if bev_map_detached.ndim != 4 or bev_map_detached.shape[0] != 1:
        raise ValueError(f"bev_map_detached must be [1,Z,Y,X], got {tuple(bev_map_detached.shape)}")
    if residual_map.ndim != 4 or residual_map.shape[0] != 1:
        raise ValueError(f"residual_map must be [1,Z,Y,X], got {tuple(residual_map.shape)}")
    if ct_bev.ndim != 4 or ct_bev.shape[0] != 1:
        raise ValueError(f"ct_bev must be [1,Z,Y,X], got {tuple(ct_bev.shape)}")
    if mlc_bev.ndim != 4 or mlc_bev.shape[0] != 1:
        raise ValueError(f"mlc_bev must be [1,Z,Y,X], got {tuple(mlc_bev.shape)}")
    if dose_raw.ndim != 3:
        raise ValueError(f"dose_raw must be [Z,Y,X], got {tuple(dose_raw.shape)}")

    chunk_z = int(RAW_FULL_OVERLAP_CHUNK_Z)
    if chunk_z <= 0:
        raise ValueError("RAW_FULL_OVERLAP_CHUNK_Z must be positive.")

    raw_z, raw_y, raw_x = (int(v) for v in dose_raw.shape)
    ref_max = torch.clamp(torch.max(dose_raw), min=1e-12)
    y = torch.arange(raw_y, dtype=residual_map.dtype, device=residual_map.device)
    x = torch.arange(raw_x, dtype=residual_map.dtype, device=residual_map.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    zero = residual_map.sum() * 0.0
    global_abs_sum = zero
    mlc_full_abs_sum = zero
    mlc_penumbra_abs_sum = zero
    outside_low_abs_sum = zero
    dose_region_abs_sum = zero
    dose_region_10_30_abs_sum = zero
    dose_region_30_80_abs_sum = zero
    dose_region_80_100_abs_sum = zero
    dose_region_squared_sum = zero
    squared_sum = zero
    residual_abs_sum = zero
    negative_point_sum = zero
    negative_value_count = zero
    negative_value_hist = (
        torch.zeros(
            (int(NEGATIVE_VALUE_HISTOGRAM_BINS),),
            dtype=residual_map.dtype,
            device=residual_map.device,
        )
        if compute_negative_value_monitor
        else None
    )

    global_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    mlc_full_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    mlc_penumbra_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    outside_low_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    dose_region_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    dose_region_10_30_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    dose_region_30_80_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    dose_region_80_100_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    body_count = torch.zeros((), dtype=residual_map.dtype, device=residual_map.device)
    pred_min = torch.full((), float("inf"), dtype=residual_map.dtype, device=residual_map.device)
    pred_negative_min = torch.full((), float("inf"), dtype=residual_map.dtype, device=residual_map.device)
    pred_max = torch.full((), float("-inf"), dtype=residual_map.dtype, device=residual_map.device)
    residual_min = torch.full((), float("inf"), dtype=residual_map.dtype, device=residual_map.device)
    residual_max = torch.full((), float("-inf"), dtype=residual_map.dtype, device=residual_map.device)

    for z0 in range(0, raw_z, chunk_z):
        z1 = min(z0 + chunk_z, raw_z)
        sampled, valid = sample_trilinear_bev_maps_chunk(
            [bev_map_detached, residual_map, ct_bev, mlc_bev],
            z0,
            z1,
            raw_y,
            raw_x,
            affine,
            yy,
            xx,
        )
        base_raw_points = sampled[0].reshape(-1)
        residual_raw_points = sampled[1].reshape(-1)
        ct_raw_points = sampled[2].reshape(-1)
        mlc_raw_points = sampled[3].reshape(-1)
        pred_raw_points = base_raw_points + residual_raw_points
        valid_flat = valid.reshape(-1)
        label_raw_points = dose_raw[z0:z1].reshape(-1).to(dtype=pred_raw_points.dtype)

        if RAW_FULL_OVERLAP_CHECK_FINITE_PER_CHUNK:
            if not torch.isfinite(pred_raw_points).all() or not torch.isfinite(label_raw_points).all():
                raise RuntimeError("NaN/Inf encountered in patient-grid predictions or labels.")

        pred_raw_points_for_mae = leaky_clamp_prediction_for_mae(pred_raw_points)
        diff = pred_raw_points_for_mae - label_raw_points
        abs_diff = torch.abs(diff)
        valid_f = valid_flat.to(dtype=residual_map.dtype)
        body = valid_flat & (ct_raw_points > float(PATIENT_GRID_CT_BODY_THRESHOLD))
        mlc_full = body & (mlc_raw_points >= 1.0 - float(PATIENT_GRID_MLC_FULL_EPS))
        mlc_penumbra = (
            body
            & (mlc_raw_points > float(PATIENT_GRID_MLC_ZERO_EPS))
            & (mlc_raw_points < 1.0 - float(PATIENT_GRID_MLC_FULL_EPS))
        )
        outside_low = (
            body
            & (mlc_raw_points <= float(PATIENT_GRID_MLC_ZERO_EPS))
            & (label_raw_points > float(PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION) * ref_max)
        )
        dose_region_10_30 = (
            valid_flat
            & (label_raw_points >= 0.10 * ref_max)
            & (label_raw_points < 0.30 * ref_max)
        )
        dose_region_30_80 = (
            valid_flat
            & (label_raw_points >= 0.30 * ref_max)
            & (label_raw_points < 0.80 * ref_max)
        )
        dose_region_80_100 = valid_flat & (label_raw_points >= 0.80 * ref_max)

        dose_region_10_30_f = dose_region_10_30.to(dtype=residual_map.dtype)
        dose_region_30_80_f = dose_region_30_80.to(dtype=residual_map.dtype)
        dose_region_80_100_f = dose_region_80_100.to(dtype=residual_map.dtype)
        mlc_full_f = mlc_full.to(dtype=residual_map.dtype)
        mlc_penumbra_f = mlc_penumbra.to(dtype=residual_map.dtype)
        outside_low_f = outside_low.to(dtype=residual_map.dtype)
        body_f = body.to(dtype=residual_map.dtype)

        dose_region_10_30_abs = torch.sum(abs_diff * dose_region_10_30_f)
        dose_region_30_80_abs = torch.sum(abs_diff * dose_region_30_80_f)
        dose_region_80_100_abs = torch.sum(abs_diff * dose_region_80_100_f)
        dose_region_10_30_n = torch.sum(dose_region_10_30_f)
        dose_region_30_80_n = torch.sum(dose_region_30_80_f)
        dose_region_80_100_n = torch.sum(dose_region_80_100_f)

        global_abs_sum = global_abs_sum + torch.sum(abs_diff * valid_f)
        mlc_full_abs_sum = mlc_full_abs_sum + torch.sum(abs_diff * mlc_full_f)
        mlc_penumbra_abs_sum = mlc_penumbra_abs_sum + torch.sum(abs_diff * mlc_penumbra_f)
        outside_low_abs_sum = outside_low_abs_sum + torch.sum(abs_diff * outside_low_f)
        diff_monitor = diff.detach()
        dose_region_f_monitor = dose_region_10_30_f + dose_region_30_80_f + dose_region_80_100_f
        squared_sum = squared_sum + torch.sum(diff_monitor * diff_monitor * valid_f)
        residual_abs_sum = residual_abs_sum + torch.sum(torch.abs(residual_raw_points.detach()) * valid_f)
        dose_region_abs_sum = dose_region_abs_sum + dose_region_10_30_abs + dose_region_30_80_abs + dose_region_80_100_abs
        dose_region_10_30_abs_sum = dose_region_10_30_abs_sum + dose_region_10_30_abs
        dose_region_30_80_abs_sum = dose_region_30_80_abs_sum + dose_region_30_80_abs
        dose_region_80_100_abs_sum = dose_region_80_100_abs_sum + dose_region_80_100_abs
        dose_region_squared_sum = dose_region_squared_sum + torch.sum(diff_monitor * diff_monitor * dose_region_f_monitor)
        global_count = global_count + torch.sum(valid_f)
        mlc_full_count = mlc_full_count + torch.sum(mlc_full_f)
        mlc_penumbra_count = mlc_penumbra_count + torch.sum(mlc_penumbra_f)
        outside_low_count = outside_low_count + torch.sum(outside_low_f)
        dose_region_count = dose_region_count + dose_region_10_30_n + dose_region_30_80_n + dose_region_80_100_n
        dose_region_10_30_count = dose_region_10_30_count + dose_region_10_30_n
        dose_region_30_80_count = dose_region_30_80_count + dose_region_30_80_n
        dose_region_80_100_count = dose_region_80_100_count + dose_region_80_100_n
        body_count = body_count + torch.sum(body_f)
        if compute_negative_value_monitor:
            if negative_value_hist is None:
                raise RuntimeError("negative_value_hist is unexpectedly None")
            negative_value_hist, chunk_negative_count, chunk_negative_min = negative_value_histogram_update(
                negative_value_hist,
                pred_raw_points,
                valid_flat,
            )
            negative_point_sum = negative_point_sum + chunk_negative_count
            negative_value_count = negative_value_count + chunk_negative_count
            pred_negative_min = torch.minimum(pred_negative_min, chunk_negative_min)
        pred_min = torch.minimum(pred_min, torch.min(torch.where(valid_flat, pred_raw_points.detach(), torch.full_like(pred_raw_points, float("inf")))))
        pred_max = torch.maximum(pred_max, torch.max(torch.where(valid_flat, pred_raw_points.detach(), torch.full_like(pred_raw_points, float("-inf")))))
        residual_min = torch.minimum(residual_min, torch.min(torch.where(valid_flat, residual_raw_points.detach(), torch.full_like(residual_raw_points, float("inf")))))
        residual_max = torch.maximum(residual_max, torch.max(torch.where(valid_flat, residual_raw_points.detach(), torch.full_like(residual_raw_points, float("-inf")))))

    if int(global_count.detach().cpu()) == 0:
        raise RuntimeError("No raw dose voxel centers overlap the BEV output grid.")

    global_mae = global_abs_sum / torch.clamp(global_count, min=1.0)
    mlc_full_mae = mlc_full_abs_sum / torch.clamp(mlc_full_count, min=1.0)
    mlc_penumbra_mae = mlc_penumbra_abs_sum / torch.clamp(mlc_penumbra_count, min=1.0)
    outside_low_mae = outside_low_abs_sum / torch.clamp(outside_low_count, min=1.0)
    global_mse = squared_sum / torch.clamp(global_count, min=1.0)
    dose_region_mae = dose_region_abs_sum / torch.clamp(dose_region_count, min=1.0)
    dose_region_10_30_mae = dose_region_10_30_abs_sum / torch.clamp(dose_region_10_30_count, min=1.0)
    dose_region_30_80_mae = dose_region_30_80_abs_sum / torch.clamp(dose_region_30_80_count, min=1.0)
    dose_region_80_100_mae = dose_region_80_100_abs_sum / torch.clamp(dose_region_80_100_count, min=1.0)
    dose_region_mse = dose_region_squared_sum / torch.clamp(dose_region_count, min=1.0)
    weighted_mae = (
        float(PATIENT_GRID_GLOBAL_MAE_WEIGHT) * global_mae
        + float(PATIENT_GRID_MLC_FULL_MAE_WEIGHT) * mlc_full_mae
        + float(PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT) * mlc_penumbra_mae
        + float(PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT) * outside_low_mae
        + float(PATIENT_GRID_DOSE_REGION_MAE_WEIGHT) * dose_region_mae
        + float(PATIENT_GRID_DOSE_REGION_10_30_MAE_WEIGHT) * dose_region_10_30_mae
        + float(PATIENT_GRID_DOSE_REGION_30_80_MAE_WEIGHT) * dose_region_30_80_mae
        + float(PATIENT_GRID_DOSE_REGION_80_100_MAE_WEIGHT) * dose_region_80_100_mae
    )
    residual_mean_abs = residual_abs_sum / torch.clamp(global_count, min=1.0)
    total_loss = weighted_mae

    with torch.no_grad():
        param = residual_map.detach()
        if compute_negative_value_monitor:
            if negative_value_hist is None:
                raise RuntimeError("negative_value_hist is unexpectedly None")
            negative_metrics = negative_value_histogram_metrics(
                negative_value_hist,
                negative_value_count,
                "patient_grid",
            )
            patient_negative_count = float(negative_value_count.detach().cpu())
            patient_negative_min = (
                float(pred_negative_min.detach().cpu())
                if patient_negative_count > 0
                else float("nan")
            )
            patient_negative_fraction = float((negative_point_sum / torch.clamp(global_count, min=1.0)).detach().cpu())
        else:
            negative_metrics = {
                "patient_grid_negative_value_p05": float("nan"),
                "patient_grid_negative_value_p95": float("nan"),
            }
            patient_negative_min = float("nan")
            patient_negative_fraction = float("nan")
        metrics = {
            "patient_branch_loss": float(total_loss.detach().cpu()),
            "patient_grid_weighted_mae": float(weighted_mae.detach().cpu()),
            "patient_grid_mlc_full_mae": float(mlc_full_mae.detach().cpu()),
            "patient_grid_mlc_penumbra_mae": float(mlc_penumbra_mae.detach().cpu()),
            "patient_grid_outside_low_dose_mae": float(outside_low_mae.detach().cpu()),
            "patient_grid_mlc_full_count": float(mlc_full_count.detach().cpu()),
            "patient_grid_mlc_penumbra_count": float(mlc_penumbra_count.detach().cpu()),
            "patient_grid_outside_low_dose_count": float(outside_low_count.detach().cpu()),
            "patient_grid_body_count": float(body_count.detach().cpu()),
            "patient_grid_gradient_weighted_mae": float("nan"),
            "patient_grid_gradient_weight_sum": 0.0,
            "patient_grid_global_mae": float(global_mae.detach().cpu()),
            "patient_grid_dose_region_10_30_mae": float(dose_region_10_30_mae.detach().cpu()),
            "patient_grid_dose_region_30_80_mae": float(dose_region_30_80_mae.detach().cpu()),
            "patient_grid_dose_region_80_100_mae": float(dose_region_80_100_mae.detach().cpu()),
            "patient_grid_dose_region_10_30_count": float(dose_region_10_30_count.detach().cpu()),
            "patient_grid_dose_region_30_80_count": float(dose_region_30_80_count.detach().cpu()),
            "patient_grid_dose_region_80_100_count": float(dose_region_80_100_count.detach().cpu()),
            "patient_grid_region_loss": float(total_loss.detach().cpu()),
            "global_mse": float(global_mse.detach().cpu()),
            "global_mae": float(global_mae.detach().cpu()),
            "raw_point_mse": float(global_mse.detach().cpu()),
            "raw_point_mae": float(global_mae.detach().cpu()),
            "dose_region_mse": float(dose_region_mse.detach().cpu()),
            "dose_region_mae": float(dose_region_mae.detach().cpu()),
            "raw_high_region_mae": float(dose_region_mae.detach().cpu()),
            "raw_mid_region_mae": float("nan"),
            "negative_prediction_fraction": patient_negative_fraction,
            "patient_grid_negative_prediction_fraction": patient_negative_fraction,
            "patient_grid_negative_value_min": patient_negative_min,
            **negative_metrics,
            "patient_grid_final_lowest_value": float(pred_min.detach().cpu()),
            "patient_grid_max_prediction_over_max_ref": float((pred_max / ref_max).detach().cpu()),
            "sampled_prediction_min": float(pred_min.detach().cpu()),
            "sampled_prediction_max": float(pred_max.detach().cpu()),
            "prediction_reference_max_ratio": float((pred_max / ref_max).detach().cpu()),
            "residual_raw_mean_abs": float(residual_mean_abs.detach().cpu()),
            "residual_raw_min": float(residual_min.detach().cpu()),
            "residual_raw_max": float(residual_max.detach().cpu()),
            "residual_map_min": float(torch.min(param).cpu()),
            "residual_map_max": float(torch.max(param).cpu()),
            "residual_map_mean": float(torch.mean(param).cpu()),
            "residual_map_std": float(torch.std(param).cpu()),
            "coefficient_min": float(torch.min(param).cpu()),
            "coefficient_max": float(torch.max(param).cpu()),
            "coefficient_mean": float(torch.mean(param).cpu()),
            "coefficient_std": float(torch.std(param).cpu()),
            "parameter_min": float(torch.min(param).cpu()),
            "parameter_max": float(torch.max(param).cpu()),
            "parameter_mean": float(torch.mean(param).cpu()),
            "parameter_std": float(torch.std(param).cpu()),
            "raw_loss": float(total_loss.detach().cpu()),
            "global_loss": float(global_mae.detach().cpu()),
            "dose_region_loss": float(dose_region_mae.detach().cpu()),
            "negative_loss": 0.0,
            "bev_aux_loss": 0.0,
            "bev_grid_anchor_loss": 0.0,
            "model_forward_time_s": float(forward_time_s),
        }
    return total_loss, metrics


def patient_grid_bev_baseline_for_supervision(bev_map: torch.Tensor) -> torch.Tensor:
    if DETACH_BEV_IN_PATIENT_GRID_SUPERVISION:
        return bev_map.detach()
    return bev_map


def compute_ragged_batch_loss_and_metrics(
    bev_maps: torch.Tensor,
    residual_maps: torch.Tensor,
    ct_bev_batch: torch.Tensor,
    mlc_bev_batch: torch.Tensor,
    dose_raw_list: list[torch.Tensor],
    dose_bev_batch: torch.Tensor,
    affine_list: list[torch.Tensor],
    sample_keys: list[str],
    n_points: int,
    forward_time_s: float,
    seed_base: int | None,
    compute_negative_value_monitor: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if bev_maps.shape != residual_maps.shape:
        raise ValueError(f"BEV/residual output shape mismatch: {tuple(bev_maps.shape)} vs {tuple(residual_maps.shape)}")
    if bev_maps.shape[0] != len(dose_raw_list):
        raise ValueError(f"Output batch {bev_maps.shape[0]} does not match raw target count {len(dose_raw_list)}")
    if ct_bev_batch.shape[0] != bev_maps.shape[0] or mlc_bev_batch.shape[0] != bev_maps.shape[0]:
        raise ValueError("CT/MLC BEV batch size must match output batch size.")
    supervision_mode = RAW_SUPERVISION_MODE.lower().strip()
    losses: list[torch.Tensor] = []
    metric_accumulator: dict[str, float] = {}
    for batch_idx, sample_key in enumerate(sample_keys):
        if supervision_mode == "patient_grid_mlc_ct_regions":
            bev_map_b = bev_maps[batch_idx : batch_idx + 1]
            residual_map_b = residual_maps[batch_idx : batch_idx + 1]
            patient_bev_baseline_b = patient_grid_bev_baseline_for_supervision(bev_map_b)
            bev_loss_b, bev_metrics_b = compute_bev_mlc_ct_region_losses_and_metrics(
                bev_map_b,
                ct_bev_batch[batch_idx : batch_idx + 1],
                mlc_bev_batch[batch_idx : batch_idx + 1],
                dose_bev_batch[batch_idx],
            )
            patient_loss_b, patient_metrics_b = compute_patient_grid_residual_losses_and_metrics(
                patient_bev_baseline_b,
                residual_map_b,
                ct_bev_batch[batch_idx : batch_idx + 1],
                mlc_bev_batch[batch_idx : batch_idx + 1],
                dose_raw_list[batch_idx],
                affine_list[batch_idx],
                forward_time_s=forward_time_s,
                compute_negative_value_monitor=compute_negative_value_monitor,
            )
            idd_bev_loss_b = bev_loss_b * 0.0
            if float(LAMBDA_IDD_BEV) != 0.0:
                idd_bev_body_mask = bev_body_mask_from_ct(ct_bev_batch[batch_idx : batch_idx + 1])
                idd_bev_loss_b = compute_bev_idd_rmse_loss(
                    bev_map_b,
                    dose_bev_batch[batch_idx],
                    idd_bev_body_mask,
                )
            idd_final_loss_b = patient_loss_b * 0.0
            if float(LAMBDA_IDD_FINAL) != 0.0:
                idd_final_loss_b = compute_final_roundtrip_idd_loss(
                    patient_bev_baseline_b,
                    residual_map_b,
                    ct_bev_batch[batch_idx : batch_idx + 1],
                    dose_raw_list[batch_idx],
                    affine_list[batch_idx],
                )
            loss_b = (
                float(BEV_BRANCH_LOSS_WEIGHT) * bev_loss_b
                + float(PATIENT_GRID_BRANCH_LOSS_WEIGHT) * patient_loss_b
                + float(LAMBDA_IDD_BEV) * idd_bev_loss_b
                + float(LAMBDA_IDD_FINAL) * idd_final_loss_b
            )
            metrics_b = {
                **patient_metrics_b,
                **bev_metrics_b,
                "idd_bev_loss": float(idd_bev_loss_b.detach().cpu()),
                "idd_final_loss": float(idd_final_loss_b.detach().cpu()),
                "idd_bev_weighted_loss": float((float(LAMBDA_IDD_BEV) * idd_bev_loss_b).detach().cpu()),
                "idd_final_weighted_loss": float((float(LAMBDA_IDD_FINAL) * idd_final_loss_b).detach().cpu()),
                "total_loss": float(loss_b.detach().cpu()),
                "bev_patient_loss_ratio": float((bev_loss_b / torch.clamp(patient_loss_b.detach(), min=1e-12)).detach().cpu()),
            }
        elif supervision_mode == "full_overlap":
            loss_b, metrics_b = compute_full_overlap_losses_and_metrics(
                bev_maps[batch_idx : batch_idx + 1] + residual_maps[batch_idx : batch_idx + 1],
                dose_raw_list[batch_idx],
                dose_bev_batch[batch_idx],
                affine_list[batch_idx],
                spline_time_s=0.0,
                forward_time_s=forward_time_s,
            )
        elif supervision_mode == "sampled_points":
            generator = torch.Generator(device=bev_maps.device)
            if seed_base is None:
                generator.seed()
            else:
                generator.manual_seed(stable_sample_seed(sample_key, seed_base))
            raw_indices_zyx = sample_raw_training_points(dose_raw_list[batch_idx], n_points, generator)
            loss_b, metrics_b = compute_losses_and_metrics(
                bev_maps[batch_idx : batch_idx + 1] + residual_maps[batch_idx : batch_idx + 1],
                dose_raw_list[batch_idx],
                dose_bev_batch[batch_idx],
                affine_list[batch_idx],
                raw_indices_zyx,
                spline_time_s=0.0,
                forward_time_s=forward_time_s,
            )
        else:
            raise ValueError(f"Unknown RAW_SUPERVISION_MODE={RAW_SUPERVISION_MODE!r}")
        losses.append(loss_b)
        add_to_accumulator(metric_accumulator, metrics_b)
    return torch.stack(losses).mean(), average_accumulator(metric_accumulator, len(losses))


def train_one_epoch(model, dataloader, device, writer, global_step: int, epoch_index: int, optimizer, ema_model: ExponentialMovingAverage | None = None):
    model.train()
    epoch_loss = 0.0
    metric_accumulator = empty_metric_accumulator()
    grad_norm_values: list[float] = []
    sample_count = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch_index}/{EPOCHS} train", leave=False)
    for batch in progress:
        batch = ensure_batched(batch)
        sample_keys = [str(value) for value in batch["sample_key"]]  # type: ignore[index]
        model_inputs, aperture_features, angle_features, dose_raw_list, dose_bev, affine_list = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        if MEASURE_STEP_TIMING_WITH_CUDA_SYNC and device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_tic = time.perf_counter()
        bev_maps, residual_maps = model(*model_inputs, aperture_features=aperture_features, angle_features=angle_features)
        bev_maps = apply_spline_output_constraints(bev_maps)
        residual_maps = apply_spline_output_constraints(residual_maps)
        if MEASURE_STEP_TIMING_WITH_CUDA_SYNC and device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_time_s = time.perf_counter() - forward_tic

        if not torch.isfinite(bev_maps).all() or not torch.isfinite(residual_maps).all():
            raise RuntimeError(f"NaN/Inf model outputs for batch {sample_keys}")
        if MEASURE_STEP_TIMING_WITH_CUDA_SYNC and device.type == "cuda":
            torch.cuda.synchronize(device)
        spline_tic = time.perf_counter()
        compute_negative_value_monitor = (
            TRAIN_NEGATIVE_VALUE_MONITOR_EVERY_STEPS <= 1
            or global_step % int(TRAIN_NEGATIVE_VALUE_MONITOR_EVERY_STEPS) == 0
        )
        loss, metrics = compute_ragged_batch_loss_and_metrics(
            bev_maps,
            residual_maps,
            model_inputs[0],
            model_inputs[1],
            dose_raw_list,
            dose_bev,
            affine_list,
            sample_keys,
            RAW_TRAIN_POINTS,
            forward_time_s=forward_time_s,
            seed_base=None,
            compute_negative_value_monitor=compute_negative_value_monitor,
        )
        if CONTRIBUTION_MONITOR_EVERY_STEPS <= 1 or global_step % int(CONTRIBUTION_MONITOR_EVERY_STEPS) == 0:
            film_contributions = film_contribution_metrics(model, aperture_features, angle_features, ",".join(sample_keys))
            skip_contributions = skip_contribution_metrics(model)
        else:
            film_contributions = {}
            skip_contributions = {}
        if MEASURE_STEP_TIMING_WITH_CUDA_SYNC and device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics["spline_sampling_time_s"] = time.perf_counter() - spline_tic
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss for batch {sample_keys}: {metrics}")
        loss.backward()
        if USE_GRAD_CLIP:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            grad_norm_value = float(grad_norm.detach().cpu())
            grad_norm_values.append(grad_norm_value)
            if GRAD_CLIP_SUMMARY_EVERY and len(grad_norm_values) % GRAD_CLIP_SUMMARY_EVERY == 0:
                window = np.asarray(grad_norm_values[-GRAD_CLIP_SUMMARY_EVERY:], dtype=np.float64)
                clipped_window = window > GRAD_CLIP_MAX_NORM
                writer.add_scalar("train_grad_window/grad_norm_mean", float(np.mean(window)), global_step)
                writer.add_scalar("train_grad_window/grad_norm_median", float(np.median(window)), global_step)
                writer.add_scalar("train_grad_window/grad_norm_p95", float(np.percentile(window, 95)), global_step)
                writer.add_scalar("train_grad_window/grad_norm_p99", float(np.percentile(window, 99)), global_step)
                writer.add_scalar("train_grad_window/grad_norm_max", float(np.max(window)), global_step)
                writer.add_scalar("train_grad_window/grad_clip_ratio", float(np.mean(clipped_window)), global_step)
        optimizer.step()
        if ema_model is not None:
            ema_model.update(model)

        loss_value = float(loss.detach().cpu())
        current_batch_size = len(sample_keys)
        sample_count += current_batch_size
        epoch_loss += loss_value * current_batch_size
        add_to_accumulator(metric_accumulator, epoch_accumulator_metrics(metrics), weight=current_batch_size)
        add_curated_training_scalars(writer, "train", loss_value, metrics, global_step)
        add_contribution_scalars(writer, "train", film_contributions, skip_contributions, global_step)
        global_step += 1
        progress.set_postfix(loss=f"{loss_value:.6g}", mae=f"{metrics['raw_point_mae']:.6g}")
    epoch_metrics = average_accumulator(metric_accumulator, sample_count)
    if grad_norm_values:
        grad_norm_array = np.asarray(grad_norm_values, dtype=np.float64)
        clipped = grad_norm_array > GRAD_CLIP_MAX_NORM
        grad_metrics = {
            "grad_norm_median": float(np.median(grad_norm_array)),
            "grad_norm_p95": float(np.percentile(grad_norm_array, 95)),
            "grad_norm_p99": float(np.percentile(grad_norm_array, 99)),
            "grad_norm_max": float(np.max(grad_norm_array)),
            "grad_clip_ratio": float(np.mean(clipped)),
        }
        epoch_metrics.update(grad_metrics)
        for key, value in grad_metrics.items():
            writer.add_scalar(f"train_grad_epoch/{key}", value, epoch_index)
    return epoch_loss / max(sample_count, 1), epoch_metrics, global_step


def validate_one_epoch(model, dataloader, device, writer, global_step: int, epoch_index: int):
    model.eval()
    epoch_loss = 0.0
    metric_accumulator = empty_metric_accumulator()
    sample_count = 0
    with torch.no_grad():
        progress = tqdm(dataloader, desc=f"Epoch {epoch_index}/{EPOCHS} val", leave=False)
        for batch in progress:
            batch = ensure_batched(batch)
            sample_keys = [str(value) for value in batch["sample_key"]]  # type: ignore[index]
            model_inputs, aperture_features, angle_features, dose_raw_list, dose_bev, affine_list = batch_to_device(batch, device)

            if MEASURE_STEP_TIMING_WITH_CUDA_SYNC and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_tic = time.perf_counter()
            bev_maps, residual_maps = model(*model_inputs, aperture_features=aperture_features, angle_features=angle_features)
            bev_maps = apply_spline_output_constraints(bev_maps)
            residual_maps = apply_spline_output_constraints(residual_maps)
            if MEASURE_STEP_TIMING_WITH_CUDA_SYNC and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_time_s = time.perf_counter() - forward_tic

            spline_tic = time.perf_counter()
            loss, metrics = compute_ragged_batch_loss_and_metrics(
                bev_maps,
                residual_maps,
                model_inputs[0],
                model_inputs[1],
                dose_raw_list,
                dose_bev,
                affine_list,
                sample_keys,
                RAW_VAL_POINTS,
                forward_time_s=forward_time_s,
                seed_base=RAW_VAL_SEED,
                compute_negative_value_monitor=True,
            )
            if CONTRIBUTION_MONITOR_EVERY_STEPS <= 1 or global_step % int(CONTRIBUTION_MONITOR_EVERY_STEPS) == 0:
                film_contributions = film_contribution_metrics(model, aperture_features, angle_features, ",".join(sample_keys))
                skip_contributions = skip_contribution_metrics(model)
            else:
                film_contributions = {}
                skip_contributions = {}
            if MEASURE_STEP_TIMING_WITH_CUDA_SYNC and device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics["spline_sampling_time_s"] = time.perf_counter() - spline_tic
            loss_value = float(loss.detach().cpu())
            current_batch_size = len(sample_keys)
            sample_count += current_batch_size
            epoch_loss += loss_value * current_batch_size
            add_to_accumulator(metric_accumulator, metrics, weight=current_batch_size)
            add_curated_training_scalars(writer, "val", loss_value, metrics, global_step)
            add_contribution_scalars(writer, "val", film_contributions, skip_contributions, global_step)
            global_step += 1
            progress.set_postfix(loss=f"{loss_value:.6g}", mae=f"{metrics['raw_point_mae']:.6g}")
    return epoch_loss / max(sample_count, 1), average_accumulator(metric_accumulator, sample_count), global_step


def run_full_raw_validation_subset(model, dataset: BSplineRawDoseDataset, sample_count: int, device: torch.device) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for idx in range(min(sample_count, len(dataset))):
            batch = dataset[idx]
            model_inputs, aperture_features, angle_features, dose_raw_list, _, affine_list = batch_to_device(batch, device)
            dose_raw = dose_raw_list[0]
            affine = affine_list[0]
            bev_map, residual_map = model(*model_inputs, aperture_features=aperture_features, angle_features=angle_features)
            bev_map = apply_spline_output_constraints(bev_map)
            residual_map = apply_spline_output_constraints(residual_map)
            pred = sample_trilinear_parameter_map_volume(
                bev_map.detach() + residual_map,
                affine,
                tuple(int(v) for v in batch["raw_shape_zyx"]),  # type: ignore[index]
                chunk_z=FULL_RAW_VALIDATION_CHUNK_Z,
            )
            ref = dose_raw
            diff = pred - ref
            ref_max = torch.clamp(torch.max(ref), min=1e-12)
            foreground = ref >= 0.10 * ref_max
            metrics = {
                "full_global_mae": float(torch.mean(torch.abs(diff)).cpu()),
                "full_foreground_mae": float(torch.mean(torch.abs(diff[foreground])).cpu()) if torch.any(foreground) else float("nan"),
                "full_prediction_max_over_reference_max": float((torch.max(pred) / ref_max).cpu()),
                "full_negative_fraction": float((pred < 0).float().mean().cpu()),
            }
            for dose_percent, distance_mm in GAMMA_CRITERIA:
                metrics[gamma_metric_name(dose_percent, distance_mm)] = float("nan")
            if FULL_RAW_VALIDATION_COMPUTE_GAMMA and torch.any(foreground):
                spacing_xyz = tuple(float(v) for v in batch["raw_spacing_xyz"])  # type: ignore[index]
                if not np.allclose(spacing_xyz, spacing_xyz[0]):
                    raise RuntimeError(f"GPU gamma helper currently expects isotropic spacing, got {spacing_xyz}")
                gamma_module = load_gamma_gpu_module()
                gamma_config = gamma_module.GammaGPUConfig(
                    spacing_mm=float(spacing_xyz[0]),
                    interp_fraction=float(GAMMA_GPU_INTERP_FRACTION),
                    point_chunk_size=int(GAMMA_GPU_POINT_CHUNK_SIZE),
                    offset_chunk_size=int(GAMMA_GPU_OFFSET_CHUNK_SIZE),
                    skip_once_passed=bool(GAMMA_GPU_SKIP_ONCE_PASSED),
                    device=str(device),
                )
                pred_np = torch.clamp_min(pred, 0.0).detach().cpu().numpy().astype(np.float32, copy=False)
                ref_np = ref.detach().cpu().numpy().astype(np.float32, copy=False)
                fg_np = foreground.detach().cpu().numpy().astype(bool, copy=False)
                ref_max_value = float(ref_max.detach().cpu())
                for dose_percent, distance_mm in GAMMA_CRITERIA:
                    metrics[gamma_metric_name(dose_percent, distance_mm)] = gamma_module.gamma_pass_rate_gpu(
                        pred_np,
                        ref_np,
                        fg_np,
                        dose_percent_threshold=float(dose_percent),
                        distance_mm_threshold=float(distance_mm),
                        reference_max=ref_max_value,
                        config=gamma_config,
                    )
            rows.append(metrics)
    out: dict[str, float] = {}
    for key in rows[0] if rows else []:
        values = [row[key] for row in rows if np.isfinite(row[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def build_run_identifier() -> str:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = RUN_NAME_SUFFIX.strip().strip("_")
    return f"{current_time}_{suffix}" if suffix else current_time


def make_experiment_dirs() -> tuple[Path, Path, Path, Path, Path]:
    experiment_dir = OUTPUT_DIR or (MODEL_CODEX_ROOT / build_run_identifier())
    log_dir = experiment_dir / "log"
    model_dir = experiment_dir / "checkpoints"
    script_dir = experiment_dir / "script"
    tensorboard_dir = experiment_dir / "tensorboard"
    for path in (log_dir, model_dir, script_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).resolve(), script_dir / Path(__file__).name)
    shutil.copy(SPLINE_PATH, script_dir / SPLINE_PATH.name)
    return experiment_dir, log_dir, model_dir, script_dir, tensorboard_dir


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"epoch_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def resolve_resume_checkpoint_path() -> Path | None:
    if not RESUME_FROM_CHECKPOINT:
        return None
    if RESUME_CHECKPOINT_PATH is not None:
        checkpoint_path = RESUME_CHECKPOINT_PATH
    else:
        if not RESUME_RUN_ID:
            raise ValueError("RESUME_FROM_CHECKPOINT=True requires RESUME_RUN_ID or RESUME_CHECKPOINT_PATH.")
        checkpoint_dir = MODEL_CODEX_ROOT / RESUME_RUN_ID / "checkpoints"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(checkpoint_dir)
        if RESUME_CHECKPOINT_EPOCH is not None:
            checkpoint_path = checkpoint_dir / f"epoch_{RESUME_CHECKPOINT_EPOCH:04d}.pt"
        elif RESUME_CHECKPOINT_NAME is not None:
            checkpoint_path = checkpoint_dir / RESUME_CHECKPOINT_NAME
        else:
            epoch_checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=checkpoint_sort_key)
            checkpoint_path = epoch_checkpoints[-1] if epoch_checkpoints else checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path


def set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def validate_resume_checkpoint_config(checkpoint: dict[str, object], checkpoint_path: Path) -> None:
    required_matches = {
        "output_representation": OUTPUT_REPRESENTATION,
        "spatial_resampler": SPATIAL_RESAMPLER,
        "resampler_align_corners": RESAMPLER_ALIGN_CORNERS,
        "model_architecture": MODEL_ARCHITECTURE,
        "input_channels": model_input_channel_count(),
        "use_inverse_square_channel": USE_INVERSE_SQUARE_CHANNEL,
        "use_wed_channel": USE_WED_CHANNEL,
        "use_aperture_film": USE_APERTURE_FILM,
        "use_angle_film": USE_ANGLE_FILM,
        "use_skip50": USE_SKIP50,
        "use_skip100": USE_SKIP100,
        "encoder_downsample_mode": ENCODER_DOWNSAMPLE_MODE,
        "output_head_mode": OUTPUT_HEAD_MODE,
    }
    legacy_resume_defaults = {
        "use_skip50": True,
        "use_skip100": True,
        "encoder_downsample_mode": "stride_conv",
        "output_head_mode": "separate_standard",
    }
    mismatches: list[str] = []
    missing: list[str] = []
    for key, expected in required_matches.items():
        if key not in checkpoint:
            if key in legacy_resume_defaults and legacy_resume_defaults[key] == expected:
                continue
            missing.append(key)
            continue
        actual = checkpoint[key]
        if isinstance(expected, float):
            matches = np.isclose(float(actual), expected, rtol=1e-7, atol=1e-8)
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{key}: checkpoint={actual!r}, current={expected!r}")
    if missing or mismatches:
        raise RuntimeError(
            "Resume checkpoint config is incompatible with the current training code. "
            f"checkpoint={checkpoint_path}; missing={missing}; mismatches={mismatches}. "
            "Use LOAD_PRETRAINED_MODEL for intentional architecture-compatible transfer, "
            "or resume from a checkpoint created by the same output/loss configuration."
        )


def load_resume_state(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    device: torch.device,
    train_batches_per_epoch: int,
    val_batches_per_epoch: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Resume checkpoint must be a dict payload, got {type(checkpoint)} from {checkpoint_path}")
    validate_resume_checkpoint_config(checkpoint, checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Resume checkpoint model_state_dict is incompatible with the current model. "
            f"missing={list(incompatible.missing_keys)}; unexpected={list(incompatible.unexpected_keys)}"
        )

    optimizer_restored = False
    scheduler_restored = False
    if RESUME_RESTORE_OPTIMIZER:
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(f"{checkpoint_path} has no optimizer_state_dict")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_restored = True
        if RESUME_OVERRIDE_OPTIMIZER_LR:
            set_optimizer_learning_rate(optimizer, LEARNING_RATE)

    if RESUME_RESTORE_SCHEDULER:
        if "scheduler_state_dict" not in checkpoint:
            raise KeyError(f"{checkpoint_path} has no scheduler_state_dict")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scheduler_restored = True

    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    train_global_step = int(
        checkpoint.get("train_global_step", checkpoint_epoch * train_batches_per_epoch)
    )
    val_global_step = int(
        checkpoint.get("val_global_step", checkpoint_epoch * val_batches_per_epoch)
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "start_epoch": checkpoint_epoch + 1,
        "train_global_step": train_global_step,
        "val_global_step": val_global_step,
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        "optimizer_restored": optimizer_restored,
        "scheduler_restored": scheduler_restored,
        "optimizer_lr": float(optimizer.param_groups[0]["lr"]),
        "scheduler_state": scheduler.state_dict(),
    }


def load_pretrained_model(model, checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model_keys = model.state_dict()
    loaded_keys = [key for key in state_dict if key in model_keys and model_keys[key].shape == state_dict[key].shape]
    loaded_params = sum(model_keys[key].numel() for key in loaded_keys)
    total_params = sum(param.numel() for param in model.parameters())
    allowed_missing_prefixes = (
        "aperture_condition_encoder.",
        "angle_condition_encoder.",
        "film64.",
        "film32.",
        "film16.",
        "decoder.film32.",
        "decoder.film16.",
    )
    bad_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Pretrained checkpoint did not load cleanly into the unchanged backbone. "
            f"bad_missing={bad_missing}; unexpected={list(incompatible.unexpected_keys)}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "loaded_parameter_count": int(loaded_params),
        "total_parameter_count": int(total_params),
        "loaded_fraction": float(loaded_params / max(total_params, 1)),
    }


def film_config_dict(aperture_stats: ApertureNormalizationStats | None = None) -> dict[str, object]:
    config = {
        "use_aperture_film": USE_APERTURE_FILM,
        "use_angle_film": USE_ANGLE_FILM,
        "aperture_condition_input_dim": 2,
        "aperture_condition_embedding_dim": 32,
        "angle_condition_input_type": ANGLE_CONDITION_INPUT_TYPE,
        "angle_condition_input_dim": 2,
        "angle_condition_hidden_dim": 16,
        "angle_condition_embedding_dim": 32,
        "film_at_tcn_output": True,
        "film_at_decoder_50": True,
        "film_at_decoder_100": True,
        "film_type": "residual",
        "film_zero_initialized": True,
    }
    if aperture_stats is not None:
        config.update(aperture_stats.to_dict())
    return config


def traditional_loss_combo_config_dict() -> dict[str, object]:
    return {
        "name": TRADITIONAL_LOSS_COMBO_NAME,
        "raw_supervision_mode": RAW_SUPERVISION_MODE,
        "bev_branch": {
            "global_mae_weight": PATIENT_GRID_GLOBAL_MAE_WEIGHT,
            "mlc_full_mae_weight": PATIENT_GRID_MLC_FULL_MAE_WEIGHT,
            "mlc_penumbra_mae_weight": PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT,
            "outside_low_dose_mae_weight": PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT,
            "dose_region_mae_weight": 0.0,
            "ct_body_threshold": PATIENT_GRID_CT_BODY_THRESHOLD,
            "mlc_zero_eps": PATIENT_GRID_MLC_ZERO_EPS,
            "mlc_full_eps": PATIENT_GRID_MLC_FULL_EPS,
            "outside_low_dose_ref_fraction": PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION,
        },
        "patient_residual_branch": {
            "global_mae_weight": PATIENT_GRID_GLOBAL_MAE_WEIGHT,
            "mlc_full_mae_weight": PATIENT_GRID_MLC_FULL_MAE_WEIGHT,
            "mlc_penumbra_mae_weight": PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT,
            "outside_low_dose_mae_weight": PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT,
            "dose_region_mae_weight": PATIENT_GRID_DOSE_REGION_MAE_WEIGHT,
            "dose_region_10_30_mae_weight": PATIENT_GRID_DOSE_REGION_10_30_MAE_WEIGHT,
            "dose_region_30_80_mae_weight": PATIENT_GRID_DOSE_REGION_30_80_MAE_WEIGHT,
            "dose_region_80_100_mae_weight": PATIENT_GRID_DOSE_REGION_80_100_MAE_WEIGHT,
            "uses_detached_bev_baseline": DETACH_BEV_IN_PATIENT_GRID_SUPERVISION,
            "patient_grid_supervision_backpropagates_to_bev_output": not DETACH_BEV_IN_PATIENT_GRID_SUPERVISION,
            "ct_body_threshold": PATIENT_GRID_CT_BODY_THRESHOLD,
            "mlc_zero_eps": PATIENT_GRID_MLC_ZERO_EPS,
            "mlc_full_eps": PATIENT_GRID_MLC_FULL_EPS,
            "outside_low_dose_ref_fraction": PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION,
        },
        "legacy_raw_overlap": {
            "global_loss_type": RAW_GLOBAL_LOSS_TYPE,
            "dose_region_loss_type": RAW_DOSE_REGION_LOSS_TYPE,
            "global_loss_weight": RAW_GLOBAL_LOSS_WEIGHT,
            "dose_region_loss_weight": RAW_DOSE_REGION_LOSS_WEIGHT,
            "dose_region_threshold": DOSE_REGION_THRESHOLD,
            "negative_dose_loss_weight": NEGATIVE_DOSE_LOSS_WEIGHT,
            "bev_aux_loss_weight": BEV_AUX_LOSS_WEIGHT,
            "bev_grid_anchor_loss_enabled": USE_BEV_GRID_ANCHOR_LOSS,
            "bev_grid_anchor_loss_weight": BEV_GRID_ANCHOR_LOSS_WEIGHT,
            "bev_grid_penumbra_loss_weight": BEV_GRID_PENUMBRA_LOSS_WEIGHT,
            "bev_grid_dose_region_loss_weight": BEV_GRID_DOSE_REGION_LOSS_WEIGHT,
        },
    }


def dose_peak_bin_weighted_mae_combo_config_dict() -> dict[str, object]:
    return {
        "name": LOSS_COMBO_NAME,
        "raw_supervision_mode": RAW_SUPERVISION_MODE,
        "region_fractions_of_beam_peak": list(DOSE_PEAK_REGION_FRACTIONS),
        "global_mae_weight": DOSE_PEAK_GLOBAL_MAE_WEIGHT,
        "region_weights": list(DOSE_PEAK_REGION_WEIGHTS),
        "normalization": "region-balanced scalar sum: global_mae + mae[1,10)% + mae[10,30)% + mae[30,80)% + mae>=80%",
        "gradient_weighted_mae_weight": GRADIENT_WEIGHTED_MAE_WEIGHT,
        "gradient_weighted_mae_min_dose_fraction": GRADIENT_WEIGHTED_MAE_MIN_DOSE_FRACTION,
        "gradient_weighted_mae_eps": GRADIENT_WEIGHTED_MAE_EPS,
        "gradient_weighted_mae_definition": "sum(M*G*abs_error)/(sum(M*G)+eps), M=GT>fraction*peak, G=|gradient(GT)| computed before masking",
        "bev_branch_target": "preprocessed_bev_reference_dose",
        "patient_branch_target": "raw_patient_grid_reference_dose_on_bev_overlap",
        "uses_mlc_aperture_masks_for_loss": False,
    }


def current_loss_combo_config_dict() -> dict[str, object]:
    return {
        "name": LOSS_COMBO_NAME,
        "raw_supervision_mode": RAW_SUPERVISION_MODE,
        "mae_definition": (
            "BEV branch: global_mae + mlc_full_mae + mlc_penumbra_mae + outside_low_dose_mae; "
            "patient branch: same terms + dose_region_10_30_mae + dose_region_30_80_mae + dose_region_80_100_mae"
        ),
        "global_mae_weight": PATIENT_GRID_GLOBAL_MAE_WEIGHT,
        "mlc_full_mae_weight": PATIENT_GRID_MLC_FULL_MAE_WEIGHT,
        "mlc_penumbra_mae_weight": PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT,
        "outside_low_dose_mae_weight": PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT,
        "dose_region_mae_weight": PATIENT_GRID_DOSE_REGION_MAE_WEIGHT,
        "dose_region_10_30_mae_weight": PATIENT_GRID_DOSE_REGION_10_30_MAE_WEIGHT,
        "dose_region_30_80_mae_weight": PATIENT_GRID_DOSE_REGION_30_80_MAE_WEIGHT,
        "dose_region_80_100_mae_weight": PATIENT_GRID_DOSE_REGION_80_100_MAE_WEIGHT,
        "dose_region_mae_loss_definition": "scalar-sum of region MAEs: [10,30)%peak, [30,80)%peak, >=80%peak",
        "outside_low_dose_ref_fraction": PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION,
        "ct_body_threshold": PATIENT_GRID_CT_BODY_THRESHOLD,
        "mlc_full_eps": PATIENT_GRID_MLC_FULL_EPS,
        "mlc_zero_eps": PATIENT_GRID_MLC_ZERO_EPS,
        "use_leaky_clamp_for_mae": USE_LEAKY_CLAMP_FOR_MAE,
        "leaky_clamp_negative_slope": LEAKY_CLAMP_NEGATIVE_SLOPE,
        "leaky_clamp_definition": "MAE uses pred if pred>=0 else negative_slope*pred; monitors use raw prediction",
        "gradient_weighted_mae_weight": GRADIENT_WEIGHTED_MAE_WEIGHT,
        "bev_branch_loss_weight": BEV_BRANCH_LOSS_WEIGHT,
        "patient_grid_branch_loss_weight": PATIENT_GRID_BRANCH_LOSS_WEIGHT,
        "lambda_idd_bev": LAMBDA_IDD_BEV,
        "lambda_idd_final": LAMBDA_IDD_FINAL,
        "final_idd_prediction": "clamp_min(final_patient_grid_prediction, 0) before body masking and BEV roundtrip",
        "bev_branch_target": "preprocessed_bev_reference_dose",
        "patient_branch_target": "raw_patient_grid_reference_dose_on_bev_overlap",
        "detach_bev_in_patient_grid_supervision": DETACH_BEV_IN_PATIENT_GRID_SUPERVISION,
        "patient_grid_supervision_backpropagates_to_bev_output": not DETACH_BEV_IN_PATIENT_GRID_SUPERVISION,
        "patient_branch_baseline": (
            "detach(bev_branch_output) + residual_branch_output"
            if DETACH_BEV_IN_PATIENT_GRID_SUPERVISION
            else "bev_branch_output + residual_branch_output"
        ),
    }


def config_dict(aperture_stats: ApertureNormalizationStats | None = None) -> dict[str, object]:
    return {
        **film_config_dict(aperture_stats),
        "loss_combo_name": LOSS_COMBO_NAME,
        "traditional_loss_combo": traditional_loss_combo_config_dict(),
        "dose_peak_bin_weighted_mae_combo": dose_peak_bin_weighted_mae_combo_config_dict(),
        "current_loss_combo": current_loss_combo_config_dict(),
        "dose_peak_bin_fractions": list(DOSE_PEAK_BIN_FRACTIONS),
        "dose_peak_bin_weights": list(DOSE_PEAK_BIN_WEIGHTS),
        "dose_peak_region_fractions": list(DOSE_PEAK_REGION_FRACTIONS),
        "dose_peak_region_weights": list(DOSE_PEAK_REGION_WEIGHTS),
        "dose_peak_global_mae_weight": DOSE_PEAK_GLOBAL_MAE_WEIGHT,
        "gradient_weighted_mae_weight": GRADIENT_WEIGHTED_MAE_WEIGHT,
        "gradient_weighted_mae_min_dose_fraction": GRADIENT_WEIGHTED_MAE_MIN_DOSE_FRACTION,
        "gradient_weighted_mae_eps": GRADIENT_WEIGHTED_MAE_EPS,
        "bev_branch_loss_weight": BEV_BRANCH_LOSS_WEIGHT,
        "patient_grid_branch_loss_weight": PATIENT_GRID_BRANCH_LOSS_WEIGHT,
        "lambda_idd_bev": LAMBDA_IDD_BEV,
        "lambda_idd_final": LAMBDA_IDD_FINAL,
        "detach_bev_in_patient_grid_supervision": DETACH_BEV_IN_PATIENT_GRID_SUPERVISION,
        "patient_grid_supervision_backpropagates_to_bev_output": not DETACH_BEV_IN_PATIENT_GRID_SUPERVISION,
        "idd_loss_eps": IDD_LOSS_EPS,
        "idd_patient_to_bev_chunk_z": IDD_PATIENT_TO_BEV_CHUNK_Z,
        "idd_body_ct_threshold_hu": IDD_BODY_CT_THRESHOLD_HU,
        "idd_body_ct_threshold_normalized": idd_body_ct_threshold_normalized(),
        "idd_fill_body_holes_2d": IDD_FILL_BODY_HOLES_2D,
        "idd_body_fill_mode": IDD_BODY_FILL_MODE,
        "idd_body_fill_connectivity": IDD_BODY_FILL_CONNECTIVITY,
        "idd_body_fill_max_iterations": IDD_BODY_FILL_MAX_ITERATIONS,
        "output_representation": OUTPUT_REPRESENTATION,
        "spatial_resampler": SPATIAL_RESAMPLER,
        "resampler_align_corners": RESAMPLER_ALIGN_CORNERS,
        "bev_parameter_spacing_mm": 2.0,
        "raw_dose_root": str(RAW_DOSE_ROOT),
        "raw_supervision_mode": RAW_SUPERVISION_MODE,
        "raw_full_overlap_chunk_z": RAW_FULL_OVERLAP_CHUNK_Z,
        "raw_full_overlap_boundary_eps": RAW_FULL_OVERLAP_BOUNDARY_EPS,
        "raw_full_overlap_check_finite_per_chunk": RAW_FULL_OVERLAP_CHECK_FINITE_PER_CHUNK,
        "raw_train_points": RAW_TRAIN_POINTS,
        "raw_val_points": RAW_VAL_POINTS,
        "raw_loss_type": RAW_GLOBAL_LOSS_TYPE,
        "raw_loss_weight": RAW_GLOBAL_LOSS_WEIGHT,
        "raw_global_loss_type": RAW_GLOBAL_LOSS_TYPE,
        "raw_dose_region_loss_type": RAW_DOSE_REGION_LOSS_TYPE,
        "raw_global_loss_weight": RAW_GLOBAL_LOSS_WEIGHT,
        "raw_dose_region_loss_weight": RAW_DOSE_REGION_LOSS_WEIGHT,
        "dose_region_threshold": DOSE_REGION_THRESHOLD,
        "negative_dose_loss_weight": NEGATIVE_DOSE_LOSS_WEIGHT,
        "bev_aux_loss_weight": BEV_AUX_LOSS_WEIGHT,
        "use_bev_grid_anchor_loss": USE_BEV_GRID_ANCHOR_LOSS,
        "bev_grid_anchor_loss_weight": BEV_GRID_ANCHOR_LOSS_WEIGHT,
        "bev_grid_penumbra_loss_weight": BEV_GRID_PENUMBRA_LOSS_WEIGHT,
        "bev_grid_dose_region_loss_weight": BEV_GRID_DOSE_REGION_LOSS_WEIGHT,
        "bev_grid_penumbra_min_fraction": BEV_GRID_PENUMBRA_MIN_FRACTION,
        "bev_grid_penumbra_max_fraction": BEV_GRID_PENUMBRA_MAX_FRACTION,
        "bev_grid_dose_region_min_fraction": BEV_GRID_DOSE_REGION_MIN_FRACTION,
        "patient_grid_global_mae_weight": PATIENT_GRID_GLOBAL_MAE_WEIGHT,
        "patient_grid_mlc_full_mae_weight": PATIENT_GRID_MLC_FULL_MAE_WEIGHT,
        "patient_grid_mlc_penumbra_mae_weight": PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT,
        "patient_grid_outside_low_dose_mae_weight": PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT,
        "patient_grid_dose_region_mae_weight": PATIENT_GRID_DOSE_REGION_MAE_WEIGHT,
        "patient_grid_dose_region_10_30_mae_weight": PATIENT_GRID_DOSE_REGION_10_30_MAE_WEIGHT,
        "patient_grid_dose_region_30_80_mae_weight": PATIENT_GRID_DOSE_REGION_30_80_MAE_WEIGHT,
        "patient_grid_dose_region_80_100_mae_weight": PATIENT_GRID_DOSE_REGION_80_100_MAE_WEIGHT,
        "patient_grid_ct_body_threshold": PATIENT_GRID_CT_BODY_THRESHOLD,
        "patient_grid_mlc_zero_eps": PATIENT_GRID_MLC_ZERO_EPS,
        "patient_grid_mlc_full_eps": PATIENT_GRID_MLC_FULL_EPS,
        "patient_grid_outside_low_dose_ref_fraction": PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION,
        "enforce_nonnegative_bspline_coefficients": ENFORCE_NONNEGATIVE_BSPLINE_COEFFICIENTS,
        "nonnegative_bspline_method": NONNEGATIVE_BSPLINE_METHOD,
        "nonnegative_softplus_beta": NONNEGATIVE_SOFTPLUS_BETA,
        "nonnegative_softplus_threshold": NONNEGATIVE_SOFTPLUS_THRESHOLD,
        "full_raw_validation_every": FULL_RAW_VALIDATION_EVERY,
        "full_raw_validation_compute_gamma": FULL_RAW_VALIDATION_COMPUTE_GAMMA,
        "gamma_criteria": [list(item) for item in GAMMA_CRITERIA],
        "gamma_gpu_interp_fraction": GAMMA_GPU_INTERP_FRACTION,
        "gamma_gpu_point_chunk_size": GAMMA_GPU_POINT_CHUNK_SIZE,
        "gamma_gpu_offset_chunk_size": GAMMA_GPU_OFFSET_CHUNK_SIZE,
        "gamma_gpu_skip_once_passed": GAMMA_GPU_SKIP_ONCE_PASSED,
        "preprocessed_root": str(PREPROCESSED_ROOT),
        "model_architecture": MODEL_ARCHITECTURE,
        "maps_filename": MAPS_FILENAME,
        "ct_dataset_name": CT_DATASET_NAME,
        "proj_dataset_name": PROJ_DATASET_NAME,
        "dose_dataset_name": DOSE_DATASET_NAME,
        "inverse_square_dataset_name": INVERSE_SQUARE_DATASET_NAME,
        "wed_dataset_name": WED_DATASET_NAME,
        "ct_scale": CT_SCALE,
        "proj_scale": PROJ_SCALE,
        "dose_scale": DOSE_SCALE,
        "use_inverse_square_channel": USE_INVERSE_SQUARE_CHANNEL,
        "inverse_square_source": "metadata_generated",
        "inverse_square_normalization": INVERSE_SQUARE_NORMALIZATION,
        "inverse_square_scale": INVERSE_SQUARE_SCALE,
        "inverse_square_offset": INVERSE_SQUARE_OFFSET,
        "use_wed_channel": USE_WED_CHANNEL,
        "wed_normalization": WED_NORMALIZATION,
        "wed_scale_mm_water": WED_SCALE_MM_WATER,
        "use_online_bev_si_flip_augmentation": USE_ONLINE_BEV_SI_FLIP_AUGMENTATION,
        "online_bev_si_flip_probability": ONLINE_BEV_SI_FLIP_PROBABILITY,
        "online_bev_si_flip_axis": ONLINE_BEV_SI_FLIP_AXIS,
        "input_channels": model_input_channel_count(),
        "tcn_dilations": TCN_DILATIONS,
        "tcn_kernel_sizes": TCN_KERNEL_SIZES,
        "tcn_temporal_modes": TCN_TEMPORAL_MODES,
        "tcn_dropout": TCN_DROPOUT,
        "tcn_norm": TCN_NORM,
        "decoder_type": DECODER_TYPE,
        "use_decoder_skips": USE_DECODER_SKIPS,
        "decoder_skip_gate_init": DECODER_SKIP_GATE_INIT,
        "use_skip_z_mixer": USE_SKIP_Z_MIXER,
        "skip_z_mixer_kernel": SKIP_Z_MIXER_KERNEL,
        "use_skip50": USE_SKIP50,
        "use_skip100": USE_SKIP100,
        "encoder_downsample_mode": ENCODER_DOWNSAMPLE_MODE,
        "output_head_mode": OUTPUT_HEAD_MODE,
        "use_input_channel_padding": USE_INPUT_CHANNEL_PADDING,
        "input_ct_pad_value": INPUT_CT_PAD_VALUE,
        "input_proj_pad_value": INPUT_PROJ_PAD_VALUE,
        "input_inverse_square_pad_value": INPUT_INVERSE_SQUARE_PAD_VALUE,
        "input_wed_pad_value": INPUT_WED_PAD_VALUE,
        "huber_delta": HUBER_DELTA,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "load_pretrained_model": LOAD_PRETRAINED_MODEL,
        "pretrained_checkpoint": str(PRETRAINED_CHECKPOINT) if PRETRAINED_CHECKPOINT is not None else None,
        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
        "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH) if RESUME_CHECKPOINT_PATH is not None else None,
        "resume_run_id": RESUME_RUN_ID,
        "resume_checkpoint_epoch": RESUME_CHECKPOINT_EPOCH,
        "resume_checkpoint_name": RESUME_CHECKPOINT_NAME,
        "resume_restore_optimizer": RESUME_RESTORE_OPTIMIZER,
        "resume_restore_scheduler": RESUME_RESTORE_SCHEDULER,
        "resume_override_optimizer_lr": RESUME_OVERRIDE_OPTIMIZER_LR,
        "resume_reset_best_val_loss": RESUME_RESET_BEST_VAL_LOSS,
        "use_ema_model": USE_EMA_MODEL,
        "ema_decay": EMA_DECAY,
        "use_grad_clip": USE_GRAD_CLIP,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "grad_clip_summary_every": GRAD_CLIP_SUMMARY_EVERY,
        "measure_step_timing_with_cuda_sync": MEASURE_STEP_TIMING_WITH_CUDA_SYNC,
        "contribution_monitor_every_steps": CONTRIBUTION_MONITOR_EVERY_STEPS,
        "train_negative_value_monitor_every_steps": TRAIN_NEGATIVE_VALUE_MONITOR_EVERY_STEPS,
    }


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            value = value.detach()
            shadow_value = self.shadow[key]
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().clone()
            for key, value in self.shadow.items()
        }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_global_step: int,
    val_global_step: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    aperture_stats: ApertureNormalizationStats,
    model_state_dict_override: dict[str, torch.Tensor] | None = None,
    extra_payload: dict[str, object] | None = None,
) -> None:
    payload = {
        **config_dict(aperture_stats),
        "epoch": epoch,
        "model_state_dict": model_state_dict_override if model_state_dict_override is not None else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_global_step": train_global_step,
        "val_global_step": val_global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, path)


def write_split_manifest(
    experiment_dir: Path,
    train_samples: list[Path],
    val_samples: list[Path],
    train_patients: list[str],
    val_patients: list[str],
    pretrained_info: dict[str, object] | None,
    resume_info: dict[str, object] | None,
    aperture_stats: ApertureNormalizationStats,
) -> None:
    manifest = {
        **config_dict(aperture_stats),
        "train_group": TRAIN_GROUP,
        "validation_group": VALIDATION_GROUP,
        "train_patient_id_list": TRAIN_PATIENT_ID_LIST,
        "train_patient_id_ranges": TRAIN_PATIENT_ID_RANGES,
        "train_validation_split_source": "preprocessed_train_and_validation_folders",
        "train_sample_fraction_per_patient": TRAIN_SAMPLE_FRACTION_PER_PATIENT,
        "val_sample_fraction_per_patient": VAL_SAMPLE_FRACTION_PER_PATIENT,
        "sample_subset_seed": SAMPLE_SUBSET_SEED,
        "train_patients": train_patients,
        "validation_patients": val_patients,
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(val_samples),
        "train_samples": [str(p) for p in train_samples],
        "validation_samples": [str(p) for p in val_samples],
        "pretrained_load_info": pretrained_info,
        "resume_load_info": resume_info,
    }
    with open(experiment_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    train_samples = discover_sample_dirs(TRAIN_GROUP, apply_train_patient_selection=True)
    val_samples = discover_sample_dirs(VALIDATION_GROUP, apply_train_patient_selection=False)
    train_samples = subset_samples_per_patient(train_samples, TRAIN_SAMPLE_FRACTION_PER_PATIENT, SAMPLE_SUBSET_SEED, "train")
    val_samples = subset_samples_per_patient(val_samples, VAL_SAMPLE_FRACTION_PER_PATIENT, SAMPLE_SUBSET_SEED, "val")
    train_samples, val_samples = limit_samples_for_smoke(train_samples, val_samples)
    if not train_samples or not val_samples:
        raise RuntimeError("Train or validation folder produced an empty set.")
    train_patients = sorted({patient_id_from_sample_dir(p) for p in train_samples}, key=patient_sort_key)
    val_patients = sorted({patient_id_from_sample_dir(p) for p in val_samples}, key=patient_sort_key)

    all_used_samples = train_samples + val_samples
    raw_lookup = build_raw_task_lookup(all_used_samples)
    precomp = load_precomputed_geometry()
    aperture_stats = compute_aperture_normalization_stats(train_samples)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    use_persistent_workers = PERSISTENT_WORKERS and NUM_WORKERS > 0
    use_pin_memory = PIN_MEMORY and device.type == "cuda"
    train_ds = BSplineRawDoseDataset(train_samples, raw_lookup, precomp, aperture_stats, enable_augmentation=True)
    val_ds = BSplineRawDoseDataset(val_samples, raw_lookup, precomp, aperture_stats, enable_augmentation=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_pin_memory, persistent_workers=use_persistent_workers, collate_fn=ragged_raw_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin_memory, persistent_workers=use_persistent_workers, collate_fn=ragged_raw_collate)

    experiment_dir, log_dir, model_dir, _, tensorboard_dir = make_experiment_dirs()
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_PATIENCE)

    resume_checkpoint_path = resolve_resume_checkpoint_path()
    pretrained_info = None
    resume_info = None
    best_val_loss = float("inf")
    start_epoch = 1
    train_global_step = 0
    val_global_step = 0
    if resume_checkpoint_path is not None:
        resume_info = load_resume_state(
            resume_checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
            train_batches_per_epoch=len(train_loader),
            val_batches_per_epoch=len(val_loader),
        )
        if RESUME_RESET_BEST_VAL_LOSS:
            best_val_loss = float("inf")
            resume_info["best_val_loss_reset_for_new_run"] = True
            resume_info["checkpoint_best_val_loss"] = resume_info["best_val_loss"]
            resume_info["best_val_loss"] = best_val_loss
        else:
            best_val_loss = float(resume_info["best_val_loss"])
        start_epoch = int(resume_info["start_epoch"])
        train_global_step = int(resume_info["train_global_step"])
        val_global_step = int(resume_info["val_global_step"])
    elif LOAD_PRETRAINED_MODEL:
        if PRETRAINED_CHECKPOINT is None or not PRETRAINED_CHECKPOINT.exists():
            raise FileNotFoundError(PRETRAINED_CHECKPOINT)
        pretrained_info = load_pretrained_model(model, PRETRAINED_CHECKPOINT, device)

    ema_model = ExponentialMovingAverage(model, EMA_DECAY) if USE_EMA_MODEL else None

    write_split_manifest(experiment_dir, train_samples, val_samples, train_patients, val_patients, pretrained_info, resume_info, aperture_stats)

    print(f"Preprocessed root: {PREPROCESSED_ROOT}")
    print(f"Train group:       {TRAIN_GROUP}")
    print(f"Validation group:  {VALIDATION_GROUP}")
    print(f"Raw dose root:     {RAW_DOSE_ROOT}")
    print(f"Experiment dir:    {experiment_dir}")
    print(f"Train patients:    {len(train_patients)}")
    print(f"Val patients:      {len(val_patients)}")
    print(f"Train samples:     {len(train_samples)} (fraction_per_patient={TRAIN_SAMPLE_FRACTION_PER_PATIENT})")
    print(f"Val samples:       {len(val_samples)} (fraction_per_patient={VAL_SAMPLE_FRACTION_PER_PATIENT})")
    print(f"Train patient list:{TRAIN_PATIENT_ID_LIST}")
    print(f"Train ranges:      {TRAIN_PATIENT_ID_RANGES}")
    print(f"Device:            {device}")
    print(f"Output repr:       {OUTPUT_REPRESENTATION}")
    print(f"Encoder downsample:{ENCODER_DOWNSAMPLE_MODE}")
    print(f"Output head mode:  {OUTPUT_HEAD_MODE}")
    print(f"Decoder skips:     skip50={USE_SKIP50}, skip100={USE_SKIP100}")
    print(f"EMA model:         {USE_EMA_MODEL}, decay={EMA_DECAY:g}")
    print(
        "Gradient clipping:"
        f" {USE_GRAD_CLIP}, max_norm={GRAD_CLIP_MAX_NORM:g}, summary_every={GRAD_CLIP_SUMMARY_EVERY}"
    )
    print(
        "Fast monitors:     "
        f"cuda_sync_timing={MEASURE_STEP_TIMING_WITH_CUDA_SYNC}, "
        f"contribution_every={CONTRIBUTION_MONITOR_EVERY_STEPS}, "
        f"train_negative_every={TRAIN_NEGATIVE_VALUE_MONITOR_EVERY_STEPS}"
    )
    print(f"Loss combo:        {LOSS_COMBO_NAME}")
    print(
        "Branch loss weight:"
        f" {BEV_BRANCH_LOSS_WEIGHT:g}*bev_branch + "
        f"{PATIENT_GRID_BRANCH_LOSS_WEIGHT:g}*patient_grid_branch"
    )
    print(
        "IDD aux loss:      "
        f"{LAMBDA_IDD_BEV:g}*bev_raw_idd + {LAMBDA_IDD_FINAL:g}*final_roundtrip_idd "
        f"(eps={IDD_LOSS_EPS:g}, patient_to_bev_chunk_z={IDD_PATIENT_TO_BEV_CHUNK_Z})"
    )
    print(
        "IDD body mask:     "
        f"ct>{IDD_BODY_CT_THRESHOLD_HU:g} HU "
        f"(normalized>{idd_body_ct_threshold_normalized():g}), "
        f"2d_fill={IDD_FILL_BODY_HOLES_2D}, mode={IDD_BODY_FILL_MODE}, connectivity={IDD_BODY_FILL_CONNECTIVITY}, "
        f"max_iter={IDD_BODY_FILL_MAX_ITERATIONS}"
    )
    print(
        "Online BEV SI flip:"
        f" {USE_ONLINE_BEV_SI_FLIP_AUGMENTATION}, p={ONLINE_BEV_SI_FLIP_PROBABILITY:g}, "
        f"axis={ONLINE_BEV_SI_FLIP_AXIS} ([Z,Y,X], Y=SI/MLC leaf)"
    )
    print("Aperture norm:")
    print(json.dumps(aperture_stats.to_dict(), indent=2))
    print(
        "Raw supervision:   "
        f"{RAW_SUPERVISION_MODE}"
    )
    if RAW_SUPERVISION_MODE.lower().strip() == "patient_grid_mlc_ct_regions":
        print(
            "MLC-region MAE:   "
            f"{PATIENT_GRID_GLOBAL_MAE_WEIGHT:g}*global_mae + "
            f"{PATIENT_GRID_MLC_FULL_MAE_WEIGHT:g}*mlc_full_mae + "
            f"{PATIENT_GRID_MLC_PENUMBRA_MAE_WEIGHT:g}*mlc_penumbra_mae + "
            f"{PATIENT_GRID_OUTSIDE_LOW_DOSE_MAE_WEIGHT:g}*outside_low_dose_mae + "
            f"{PATIENT_GRID_DOSE_REGION_MAE_WEIGHT:g}*dose_region_mae + "
            f"{PATIENT_GRID_DOSE_REGION_10_30_MAE_WEIGHT:g}*dose_region_10_30_mae + "
            f"{PATIENT_GRID_DOSE_REGION_30_80_MAE_WEIGHT:g}*dose_region_30_80_mae + "
            f"{PATIENT_GRID_DOSE_REGION_80_100_MAE_WEIGHT:g}*dose_region_80_100_mae"
        )
        print(
            "MLC-region masks: "
            f"body=CT>{PATIENT_GRID_CT_BODY_THRESHOLD:g}, "
            f"full=MLC>={1.0 - float(PATIENT_GRID_MLC_FULL_EPS):g}, "
            f"penumbra={PATIENT_GRID_MLC_ZERO_EPS:g}<MLC<{1.0 - float(PATIENT_GRID_MLC_FULL_EPS):g}, "
            f"outside_low=MLC<={PATIENT_GRID_MLC_ZERO_EPS:g} and ref>{PATIENT_GRID_OUTSIDE_LOW_DOSE_REF_FRACTION:g}*peak"
        )
        print(
            "Gradient MAE loss: disabled "
            f"(weight={GRADIENT_WEIGHTED_MAE_WEIGHT:g})"
        )
    elif RAW_SUPERVISION_MODE.lower().strip() == "full_overlap":
        print(
            "Raw overlap loss:  "
            f"{RAW_GLOBAL_LOSS_WEIGHT}*global_{RAW_GLOBAL_LOSS_TYPE} + "
            f"{RAW_DOSE_REGION_LOSS_WEIGHT}*dose_region_{RAW_DOSE_REGION_LOSS_TYPE}, "
            f"dose_region>={DOSE_REGION_THRESHOLD:g}*raw_max"
        )
        print(f"Raw overlap chunk: {RAW_FULL_OVERLAP_CHUNK_Z} z-slices")
    else:
        print(f"Raw sample points: train={RAW_TRAIN_POINTS}, val={RAW_VAL_POINTS}")
    print(
        "Spline nonnegative:"
        f" {ENFORCE_NONNEGATIVE_BSPLINE_COEFFICIENTS}"
        f" ({NONNEGATIVE_BSPLINE_METHOD})"
    )
    print(f"Negative loss:     {NEGATIVE_DOSE_LOSS_WEIGHT}")
    print(f"BEV aux loss:      {BEV_AUX_LOSS_WEIGHT}")
    print(
        "Patient->BEV grad: "
        f"{not DETACH_BEV_IN_PATIENT_GRID_SUPERVISION} "
        f"(detach_bev_in_patient_grid_supervision={DETACH_BEV_IN_PATIENT_GRID_SUPERVISION})"
    )
    print(
        "Res branch loss:   "
        f"{BEV_BRANCH_LOSS_WEIGHT:g}*BEV branch and "
        f"{PATIENT_GRID_BRANCH_LOSS_WEIGHT:g}*patient residual branch are supervised with traditional "
        "MLC-region MAE. Patient residual branch baseline is "
        + (
            "detach(BEV branch) + residual branch. "
            if DETACH_BEV_IN_PATIENT_GRID_SUPERVISION
            else "BEV branch + residual branch, so patient-grid losses also update the BEV head. "
        )
        + "Small IDD losses are auxiliary."
    )
    if pretrained_info is not None:
        print("Pretrained load:")
        print(json.dumps(pretrained_info, indent=2))
    if resume_info is not None:
        print("Resume load:")
        print(json.dumps(resume_info, indent=2, default=str))

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    log_file = log_dir / "run.log"
    with open(log_file, "w") as f:
        f.write("Aperture normalization:\n")
        f.write(json.dumps(aperture_stats.to_dict(), indent=2) + "\n")
        if pretrained_info is not None:
            f.write("Pretrained load:\n")
            f.write(json.dumps(pretrained_info, indent=2) + "\n")
        if resume_info is not None:
            f.write("Resume load:\n")
            f.write(json.dumps(resume_info, indent=2, default=str) + "\n")
        if start_epoch > EPOCHS:
            raise RuntimeError(
                f"Resume checkpoint epoch is {start_epoch - 1}, but EPOCHS={EPOCHS}. "
                "Increase EPOCHS to continue training."
            )
        for epoch_index in range(start_epoch, EPOCHS + 1):
            print(f"\nEpoch {epoch_index}/{EPOCHS}")
            train_loss, train_metrics, train_global_step = train_one_epoch(model, train_loader, device, writer, train_global_step, epoch_index, optimizer, ema_model)
            val_loss, val_metrics, val_global_step = validate_one_epoch(model, val_loader, device, writer, val_global_step, epoch_index)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            msg = f"Epoch {epoch_index}/{EPOCHS}, LR: {current_lr:.2e}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            print(msg)
            f.write(msg + "\n")
            add_curated_training_scalars(writer, "train_epoch", train_loss, train_metrics, epoch_index)
            add_curated_training_scalars(writer, "val_epoch", val_loss, val_metrics, epoch_index)

            if FULL_RAW_VALIDATION_EVERY and epoch_index % FULL_RAW_VALIDATION_EVERY == 0:
                full_metrics = run_full_raw_validation_subset(model, val_ds, FULL_RAW_VALIDATION_MAX_SAMPLES, device)
                add_scalars(writer, "val_full_raw_epoch", full_metrics, epoch_index)
                f.write("Full raw validation subset: " + json.dumps(full_metrics) + "\n")
                print("Full raw validation subset:", full_metrics)

            if SAVE_EACH_X_EPOCH and epoch_index % SAVE_EACH_X_EPOCH == 0:
                save_checkpoint(model_dir / f"epoch_{epoch_index:04d}.pt", model, optimizer, scheduler, epoch_index, train_global_step, val_global_step, train_loss, val_loss, best_val_loss, aperture_stats)
                if ema_model is not None:
                    save_checkpoint(
                        model_dir / f"epoch_{epoch_index:04d}_ema.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch_index,
                        train_global_step,
                        val_global_step,
                        train_loss,
                        val_loss,
                        best_val_loss,
                        aperture_stats,
                        model_state_dict_override=ema_model.state_dict(),
                        extra_payload={"ema_decay": EMA_DECAY, "ema_checkpoint": True},
                    )
            if SAVE_BEST_MODEL and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model_dir / "best_model.pt", model, optimizer, scheduler, epoch_index, train_global_step, val_global_step, train_loss, val_loss, best_val_loss, aperture_stats)
                if ema_model is not None:
                    save_checkpoint(
                        model_dir / "best_model_ema.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch_index,
                        train_global_step,
                        val_global_step,
                        train_loss,
                        val_loss,
                        best_val_loss,
                        aperture_stats,
                        model_state_dict_override=ema_model.state_dict(),
                        extra_payload={"ema_decay": EMA_DECAY, "ema_checkpoint": True},
                    )
                print("Best model saved")
                f.write("Best model saved\n")
            f.flush()
            writer.flush() 
    writer.close()


if __name__ == "__main__":
    main()
