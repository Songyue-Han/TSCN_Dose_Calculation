#!/usr/bin/env python3
"""
DoseRAD2026 BEV preprocessing with dynamic z offset and true-ray WED.

Runtime scope, output path, and grid size are controlled by hard-coded constants
near the top of this file.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm


SAMPLE_INDEX = 0
SAMPLE_KEY: str | None = None
PROCESS_ALL_SAMPLES = True
MAX_SAMPLES: int | None = None
OVERWRITE_EXISTING = False
HDF5_MAP_FILENAME = "maps.h5"
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_OPTS = 1
HDF5_SHUFFLE = True
HDF5_CHUNKS = (16, 200, 200)

DATA_ROOT = Path(__file__).resolve().parent / "data" / "DoseRAD2026"
OUTPUT_ROOT = Path(__file__).resolve().parent / "data" / "bev_online"
PRECOMPUTE_DIR = Path(__file__).resolve().parent / "precomputed_geometry_v1_2"
PATIENT_ID_START: str | None = None
PATIENT_ID_END: str | None = None

DEVICE = "cuda:0"
Z_CHUNK_SIZE = 64
INTERP_ORDER = 1
USE_GPU_RESAMPLING = True
BODY_POINT_CHUNK_SIZE = 1_000_000

OUTPUT_DEPTH = 300
MARGIN_SLICES = 5
ALLOWED_OFFSET_SLICES = 250
BODY_THRESHOLD_HU = -1020.0
CT_AIR_HU = -1024.0
OUTPUT_MAP_NAMES = ("ct", "mr", "mass_density", "wed", "proj_fraction", "dose")


def import_preprocessing_module():
    path = Path(__file__).resolve().parent / "1_Preprocessing.py"
    spec = importlib.util.spec_from_file_location("dr26_preprocessing_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRE = import_preprocessing_module()


@dataclass
class PatientGpuCache:
    patient_id: str
    ct_img: sitk.Image
    mr_img: sitk.Image | None
    mr_ref_img: sitk.Image
    ct_gpu: torch.Tensor
    mr_gpu: torch.Tensor
    density_gpu: torch.Tensor
    ct_path: Path
    mr_path: Path | None
    body_world_xyz: torch.Tensor
    body_non_air_voxel_count: int


@dataclass
class WEDGeometryGPU:
    z_centers_np: np.ndarray
    z_edges_np: np.ndarray
    x_centers: torch.Tensor
    y_centers: torch.Tensor
    ray_u: torch.Tensor
    ray_v: torch.Tensor
    metric: torch.Tensor
    uu: torch.Tensor
    vv: torch.Tensor
    xx: torch.Tensor
    yy: torch.Tensor


def load_precompute() -> dict[str, np.ndarray]:
    npz_path = PRECOMPUTE_DIR / "precompute_geometry.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Run 1.2_Precompute_Geometry.py first: {npz_path}")
    data = np.load(npz_path)
    return {name: data[name] for name in data.files}


def selected_patient_ids_from_range(training_root: Path) -> set[str] | None:
    if PATIENT_ID_START is None and PATIENT_ID_END is None:
        return None
    selected = {
        patient_dir.name
        for patient_dir in PRE.patient_dirs(training_root)
        if (PATIENT_ID_START is None or patient_dir.name >= PATIENT_ID_START)
        and (PATIENT_ID_END is None or patient_dir.name <= PATIENT_ID_END)
    }
    if not selected:
        raise ValueError(f"No patients found in configured range {PATIENT_ID_START}..{PATIENT_ID_END}")
    return selected


def load_hu_density_curve(training_root: Path) -> tuple[np.ndarray, np.ndarray]:
    params_path = training_root / "beam_parameters.json"
    with open(params_path, "r") as f:
        params = json.load(f)
    entries = params["hu_to_density"]["entries"]
    hu = np.asarray([entry["hu"] for entry in entries], dtype=np.float32)
    density = np.asarray([entry["density_g_cm3"] for entry in entries], dtype=np.float32)
    order = np.argsort(hu)
    return hu[order], density[order]


def hu_to_mass_density(ct_hu_zyx: np.ndarray, hu: np.ndarray, density: np.ndarray) -> np.ndarray:
    return np.interp(ct_hu_zyx.astype(np.float32), hu, density).astype(np.float32)


def hu_to_mass_density_torch(ct_hu_zyx: torch.Tensor, hu: np.ndarray, density: np.ndarray) -> torch.Tensor:
    hu_t = torch.as_tensor(hu, dtype=torch.float32, device=ct_hu_zyx.device)
    density_t = torch.as_tensor(density, dtype=torch.float32, device=ct_hu_zyx.device)
    idx = torch.bucketize(ct_hu_zyx, hu_t).clamp(1, hu_t.numel() - 1)
    x0 = hu_t[idx - 1]
    x1 = hu_t[idx]
    y0 = density_t[idx - 1]
    y1 = density_t[idx]
    weight = (ct_hu_zyx - x0) / torch.clamp(x1 - x0, min=1e-6)
    interpolated = y0 + weight * (y1 - y0)
    interpolated = torch.where(ct_hu_zyx <= hu_t[0], density_t[0], interpolated)
    interpolated = torch.where(ct_hu_zyx >= hu_t[-1], density_t[-1], interpolated)
    return interpolated


def build_bev_world_grid(geometry, z_centers: np.ndarray, y_centers: np.ndarray, x_centers: np.ndarray) -> np.ndarray:
    depth_axis = np.asarray(geometry.beam_axis_xyz, dtype=np.float64)
    si_axis = np.asarray(geometry.si_axis_xyz, dtype=np.float64)
    lateral_axis = np.asarray(geometry.lateral_axis_xyz, dtype=np.float64)
    iso = np.asarray(geometry.iso_center_xyz, dtype=np.float64)

    z_grid, y_grid, x_grid = np.meshgrid(z_centers, y_centers, x_centers, indexing="ij")
    return (
        iso
        + z_grid[..., None] * depth_axis
        + y_grid[..., None] * si_axis
        + x_grid[..., None] * lateral_axis
    )


def valid_mask_from_coords(coords_zyx: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    valid = np.ones(coords_zyx.shape[1], dtype=bool)
    for axis, size in enumerate(shape_zyx):
        valid &= (coords_zyx[axis] >= 0.0) & (coords_zyx[axis] <= (size - 1))
    return valid


def resample_array_to_bev_coords(
    array_zyx: np.ndarray,
    image: sitk.Image,
    geometry,
    z_centers: np.ndarray,
    y_centers: np.ndarray,
    x_centers: np.ndarray,
    order: int,
    cval: float,
    chunk_size: int = 64,
    return_valid: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    out_shape_yx = (len(y_centers), len(x_centers))
    for z0 in range(0, len(z_centers), chunk_size):
        z_chunk = z_centers[z0:z0 + chunk_size]
        world = build_bev_world_grid(geometry, z_chunk, y_centers, x_centers)
        coords = PRE.world_to_continuous_indices(image, world)
        sampled = PRE.map_coordinates_cpu(
            array_zyx,
            coords,
            order=order,
            cval=cval,
            output_shape=(len(z_chunk), *out_shape_yx),
        )
        chunks.append(sampled)
        if return_valid:
            valid_chunks.append(valid_mask_from_coords(coords, array_zyx.shape).reshape((len(z_chunk), *out_shape_yx)))
    sampled_all = np.concatenate(chunks, axis=0).astype(np.float32)
    valid_all = np.concatenate(valid_chunks, axis=0) if return_valid else None
    return sampled_all, valid_all


def image_geometry_torch(image: sitk.Image, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = torch.as_tensor(image.GetOrigin(), dtype=torch.float32, device=device)
    spacing = torch.as_tensor(image.GetSpacing(), dtype=torch.float32, device=device)
    direction = torch.as_tensor(image.GetDirection(), dtype=torch.float32, device=device).reshape(3, 3)
    size_xyz = torch.as_tensor(image.GetSize(), dtype=torch.float32, device=device)
    return origin, spacing, direction, size_xyz


def torch_volume(array_zyx: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array_zyx, dtype=torch.float32, device=device)[None, None, :, :, :]


def build_bev_grid_world_torch(
    geometry,
    z_centers: torch.Tensor,
    y_centers: torch.Tensor,
    x_centers: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    depth_axis = torch.as_tensor(geometry.beam_axis_xyz, dtype=torch.float32, device=device)
    si_axis = torch.as_tensor(geometry.si_axis_xyz, dtype=torch.float32, device=device)
    lateral_axis = torch.as_tensor(geometry.lateral_axis_xyz, dtype=torch.float32, device=device)
    iso = torch.as_tensor(geometry.iso_center_xyz, dtype=torch.float32, device=device)
    z_grid, y_grid, x_grid = torch.meshgrid(z_centers, y_centers, x_centers, indexing="ij")
    return (
        iso
        + z_grid[..., None] * depth_axis
        + y_grid[..., None] * si_axis
        + x_grid[..., None] * lateral_axis
    )


def world_to_grid_sample_torch(world_xyz: torch.Tensor, image: sitk.Image, device: torch.device) -> torch.Tensor:
    origin, spacing, direction, size_xyz = image_geometry_torch(image, device)
    continuous_xyz = ((world_xyz.reshape(-1, 3) - origin) @ direction) / spacing
    normalized = 2.0 * continuous_xyz / torch.clamp(size_xyz - 1.0, min=1.0) - 1.0
    return normalized.reshape(*world_xyz.shape[:-1], 3)


def resample_volume_to_bev_gpu(
    volume_1cdhw: torch.Tensor,
    image: sitk.Image,
    geometry,
    z_centers_mm: np.ndarray,
    y_centers_mm: np.ndarray,
    x_centers_mm: np.ndarray,
    cval: float,
    device: torch.device,
    chunk_size: int = Z_CHUNK_SIZE,
    return_numpy: bool = True,
) -> np.ndarray | torch.Tensor:
    y_t = torch.as_tensor(y_centers_mm, dtype=torch.float32, device=device)
    x_t = torch.as_tensor(x_centers_mm, dtype=torch.float32, device=device)
    outputs: list[torch.Tensor] = []
    volume_delta = volume_1cdhw - float(cval)
    with torch.inference_mode():
        for z0 in range(0, len(z_centers_mm), chunk_size):
            z_t = torch.as_tensor(z_centers_mm[z0:z0 + chunk_size], dtype=torch.float32, device=device)
            world = build_bev_grid_world_torch(geometry, z_t, y_t, x_t, device)
            grid = world_to_grid_sample_torch(world, image, device)[None, ...]
            sampled = F.grid_sample(
                volume_delta,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0] + float(cval)
            outputs.append(sampled)
    result = torch.cat(outputs, dim=0)
    if return_numpy:
        return result.detach().cpu().numpy()
    return result


def build_bev_sampling_grid_gpu(
    image: sitk.Image,
    geometry,
    z_centers_mm: np.ndarray,
    y_centers_mm: np.ndarray,
    x_centers_mm: np.ndarray,
    device: torch.device,
    chunk_size: int = Z_CHUNK_SIZE,
) -> torch.Tensor:
    y_t = torch.as_tensor(y_centers_mm, dtype=torch.float32, device=device)
    x_t = torch.as_tensor(x_centers_mm, dtype=torch.float32, device=device)
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for z0 in range(0, len(z_centers_mm), chunk_size):
            z_t = torch.as_tensor(z_centers_mm[z0:z0 + chunk_size], dtype=torch.float32, device=device)
            world = build_bev_grid_world_torch(geometry, z_t, y_t, x_t, device)
            chunks.append(world_to_grid_sample_torch(world, image, device))
    return torch.cat(chunks, dim=0)[None, ...].contiguous()


def sample_with_prebuilt_grid(
    volume_1cdhw: torch.Tensor,
    sampling_grid_1dhw3: torch.Tensor,
    cval: float,
    chunk_size: int = Z_CHUNK_SIZE,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    volume_delta = volume_1cdhw - float(cval)
    with torch.inference_mode():
        for z0 in range(0, sampling_grid_1dhw3.shape[1], chunk_size):
            grid = sampling_grid_1dhw3[:, z0:z0 + chunk_size]
            sampled = F.grid_sample(
                volume_delta,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0] + float(cval)
            outputs.append(sampled)
    return torch.cat(outputs, dim=0)


def compute_output_window_from_body_points(
    ct_gpu_zyx: torch.Tensor,
    ct_img: sitk.Image,
    geometry,
    z_centers: np.ndarray,
    device: torch.device,
) -> dict:
    non_air = torch.nonzero(ct_gpu_zyx > BODY_THRESHOLD_HU, as_tuple=False)
    if non_air.numel() == 0:
        return compute_output_window(np.zeros(len(z_centers), dtype=bool), z_centers)

    origin, spacing, direction, _ = image_geometry_torch(ct_img, device)
    iso = torch.as_tensor(geometry.iso_center_xyz, dtype=torch.float32, device=device)
    lateral_axis = torch.as_tensor(geometry.lateral_axis_xyz, dtype=torch.float32, device=device)
    si_axis = torch.as_tensor(geometry.si_axis_xyz, dtype=torch.float32, device=device)
    depth_axis = torch.as_tensor(geometry.beam_axis_xyz, dtype=torch.float32, device=device)

    z_min: torch.Tensor | None = None
    z_max: torch.Tensor | None = None
    for start in range(0, non_air.shape[0], BODY_POINT_CHUNK_SIZE):
        chunk_zyx = non_air[start:start + BODY_POINT_CHUNK_SIZE].to(device=device, dtype=torch.float32)
        index_xyz = torch.stack((chunk_zyx[:, 2], chunk_zyx[:, 1], chunk_zyx[:, 0]), dim=1)
        world = origin + (index_xyz * spacing) @ direction.T
        delta = world - iso
        x_bev = delta @ lateral_axis
        y_bev = delta @ si_axis
        valid = (torch.abs(x_bev) <= 200.0) & (torch.abs(y_bev) <= 200.0)
        if not torch.any(valid):
            continue
        z_bev = delta[valid] @ depth_axis
        local_min = z_bev.min()
        local_max = z_bev.max()
        z_min = local_min if z_min is None else torch.minimum(z_min, local_min)
        z_max = local_max if z_max is None else torch.maximum(z_max, local_max)

    if z_min is None or z_max is None:
        return compute_output_window(np.zeros(len(z_centers), dtype=bool), z_centers)

    spacing_mm = float(z_centers[1] - z_centers[0])
    first_z = float(z_centers[0])
    body_min_index = int(np.floor((float(z_min.detach().cpu()) - first_z) / spacing_mm))
    body_max_index = int(np.ceil((float(z_max.detach().cpu()) - first_z) / spacing_mm))
    body_min_index = int(np.clip(body_min_index, 0, len(z_centers) - 1))
    body_max_index = int(np.clip(body_max_index, 0, len(z_centers) - 1))
    occupied = np.zeros(len(z_centers), dtype=bool)
    occupied[body_min_index:body_max_index + 1] = True
    info = compute_output_window(occupied, z_centers)
    info["body_z_min_mm_raw"] = float(z_min.detach().cpu())
    info["body_z_max_mm_raw"] = float(z_max.detach().cpu())
    info["body_range_method"] = "gpu_raw_ct_non_air_voxel_projection"
    info["body_non_air_voxel_count"] = int(non_air.shape[0])
    return info


def build_patient_gpu_cache(
    task,
    hu: np.ndarray,
    density_curve: np.ndarray,
    device: torch.device,
) -> PatientGpuCache:
    ct_img = PRE.read_float_image(task.ct_path)
    mr_path = task.patient_dir / "image" / "mr.mha"
    mr_img = PRE.read_float_image(mr_path) if mr_path.exists() else None

    ct_array = sitk.GetArrayFromImage(ct_img).astype(np.float32, copy=False)
    mr_array = (
        sitk.GetArrayFromImage(mr_img).astype(np.float32, copy=False)
        if mr_img is not None
        else np.zeros_like(ct_array, dtype=np.float32)
    )
    ct_gpu = torch.as_tensor(ct_array, dtype=torch.float32, device=device)
    mr_gpu = torch.as_tensor(mr_array, dtype=torch.float32, device=device)
    density_gpu = hu_to_mass_density_torch(ct_gpu, hu, density_curve)

    non_air_zyx = torch.nonzero(ct_gpu > BODY_THRESHOLD_HU, as_tuple=False)
    if non_air_zyx.numel() == 0:
        body_world_xyz = torch.empty((0, 3), dtype=torch.float32, device=device)
    else:
        origin, spacing, direction, _ = image_geometry_torch(ct_img, device)
        index_xyz = torch.stack(
            (
                non_air_zyx[:, 2].to(dtype=torch.float32),
                non_air_zyx[:, 1].to(dtype=torch.float32),
                non_air_zyx[:, 0].to(dtype=torch.float32),
            ),
            dim=1,
        )
        body_world_xyz = (origin + (index_xyz * spacing) @ direction.T).contiguous()

    return PatientGpuCache(
        patient_id=task.patient_id,
        ct_img=ct_img,
        mr_img=mr_img,
        mr_ref_img=mr_img if mr_img is not None else ct_img,
        ct_gpu=ct_gpu,
        mr_gpu=mr_gpu,
        density_gpu=density_gpu,
        ct_path=task.ct_path,
        mr_path=mr_path if mr_path.exists() else None,
        body_world_xyz=body_world_xyz,
        body_non_air_voxel_count=int(non_air_zyx.shape[0]),
    )


def compute_output_window_from_patient_cache(
    patient_cache: PatientGpuCache,
    geometry,
    z_centers: np.ndarray,
    device: torch.device,
) -> dict:
    if patient_cache.body_world_xyz.numel() == 0:
        return compute_output_window(np.zeros(len(z_centers), dtype=bool), z_centers)

    iso = torch.as_tensor(geometry.iso_center_xyz, dtype=torch.float32, device=device)
    lateral_axis = torch.as_tensor(geometry.lateral_axis_xyz, dtype=torch.float32, device=device)
    si_axis = torch.as_tensor(geometry.si_axis_xyz, dtype=torch.float32, device=device)
    depth_axis = torch.as_tensor(geometry.beam_axis_xyz, dtype=torch.float32, device=device)
    basis = torch.stack((lateral_axis, si_axis, depth_axis), dim=1)

    z_min = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    z_max = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
    body_world = patient_cache.body_world_xyz
    for start in range(0, body_world.shape[0], BODY_POINT_CHUNK_SIZE):
        bev = (body_world[start:start + BODY_POINT_CHUNK_SIZE] - iso) @ basis
        valid = (torch.abs(bev[:, 0]) <= 200.0) & (torch.abs(bev[:, 1]) <= 200.0)
        z_bev = bev[:, 2]
        local_min = torch.where(valid, z_bev, torch.full_like(z_bev, float("inf"))).amin()
        local_max = torch.where(valid, z_bev, torch.full_like(z_bev, float("-inf"))).amax()
        z_min = torch.minimum(z_min, local_min)
        z_max = torch.maximum(z_max, local_max)

    bounds = torch.stack((z_min, z_max)).detach().cpu().numpy()
    if not np.isfinite(bounds).all():
        return compute_output_window(np.zeros(len(z_centers), dtype=bool), z_centers)

    spacing_mm = float(z_centers[1] - z_centers[0])
    first_z = float(z_centers[0])
    z_min_value = float(bounds[0])
    z_max_value = float(bounds[1])
    body_min_index = int(np.floor((z_min_value - first_z) / spacing_mm))
    body_max_index = int(np.ceil((z_max_value - first_z) / spacing_mm))
    body_min_index = int(np.clip(body_min_index, 0, len(z_centers) - 1))
    body_max_index = int(np.clip(body_max_index, 0, len(z_centers) - 1))
    occupied = np.zeros(len(z_centers), dtype=bool)
    occupied[body_min_index:body_max_index + 1] = True
    info = compute_output_window(occupied, z_centers)
    info["body_z_min_mm_raw"] = z_min_value
    info["body_z_max_mm_raw"] = z_max_value
    info["body_range_method"] = "gpu_cached_body_world_single_bev_matmul"
    info["body_non_air_voxel_count"] = patient_cache.body_non_air_voxel_count
    return info


def compute_output_window(occupied: np.ndarray, z_centers: np.ndarray) -> dict:
    start0 = (len(z_centers) - OUTPUT_DEPTH) // 2
    end0 = start0 + OUTPUT_DEPTH - 1
    min_start = max(0, start0 - ALLOWED_OFFSET_SLICES)
    max_start = min(len(z_centers) - OUTPUT_DEPTH, start0 + ALLOWED_OFFSET_SLICES)

    occupied_indices = np.flatnonzero(occupied)
    if occupied_indices.size == 0:
        output_start = start0
        body_min = body_max = None
        no_body = True
    else:
        body_min = int(occupied_indices[0])
        body_max = int(occupied_indices[-1])
        no_body = False
        upstream_default = body_min <= start0
        downstream_default = body_max >= end0
        if not upstream_default and not downstream_default:
            output_start = start0
        else:
            lower_start = body_max + MARGIN_SLICES - (OUTPUT_DEPTH - 1)
            upper_start = body_min - MARGIN_SLICES
            if lower_start <= upper_start:
                output_start = min(max(start0, lower_start), upper_start)
            else:
                output_start = body_min - MARGIN_SLICES
    output_start = int(min(max(output_start, min_start), max_start))
    output_end = output_start + OUTPUT_DEPTH - 1

    upstream_after = False if body_min is None else body_min <= output_start
    downstream_after = False if body_max is None else body_max >= output_end
    return {
        "no_body_found": no_body,
        "body_min_index": body_min,
        "body_max_index": body_max,
        "output_start_index": output_start,
        "output_end_index": output_end,
        "bev_z_offset_mm": float((output_start - start0) * 2.0),
        "body_z_min_mm": None if body_min is None else float(z_centers[body_min]),
        "body_z_max_mm": None if body_max is None else float(z_centers[body_max]),
        "output_z_min_mm": float(z_centers[output_start]),
        "output_z_max_mm": float(z_centers[output_end]),
        "upstream_truncated_after_offset": bool(upstream_after),
        "downstream_truncated_after_offset": bool(downstream_after),
    }


def make_shifted_bev_sitk(array_dyx: np.ndarray, geometry, z_centers: np.ndarray, y_centers: np.ndarray, x_centers: np.ndarray) -> sitk.Image:
    depth_axis = np.asarray(geometry.beam_axis_xyz, dtype=np.float64)
    si_axis = np.asarray(geometry.si_axis_xyz, dtype=np.float64)
    lateral_axis = np.asarray(geometry.lateral_axis_xyz, dtype=np.float64)
    iso = np.asarray(geometry.iso_center_xyz, dtype=np.float64)
    origin = iso + z_centers[0] * depth_axis + y_centers[0] * si_axis + x_centers[0] * lateral_axis

    image = sitk.GetImageFromArray(array_dyx.astype(np.float32))
    image.SetSpacing((2.0, 2.0, 2.0))
    image.SetOrigin(tuple(float(v) for v in origin))
    direction = np.column_stack((lateral_axis, si_axis, depth_axis))
    image.SetDirection(tuple(float(v) for v in direction.reshape(-1)))
    return image


def get_device() -> torch.device:
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: {DEVICE} requested but CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(DEVICE)


def physical_to_grid(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return 2.0 * (values - centers[0]) / (centers[-1] - centers[0]) - 1.0


def build_wed_geometry_gpu(precomp: dict[str, np.ndarray], device: torch.device) -> WEDGeometryGPU:
    z_centers_np = precomp["master_z_centers_mm"].astype(np.float32)
    z_edges_np = precomp["master_z_edges_mm"].astype(np.float32)
    x_centers = torch.as_tensor(precomp["cart_x_centers_mm"].astype(np.float32), dtype=torch.float32, device=device)
    y_centers = torch.as_tensor(precomp["cart_y_centers_mm"].astype(np.float32), dtype=torch.float32, device=device)
    ray_u = torch.as_tensor(precomp["ray_u_mm"].astype(np.float32), dtype=torch.float32, device=device)
    ray_v = torch.as_tensor(precomp["ray_v_mm"].astype(np.float32), dtype=torch.float32, device=device)
    metric = torch.as_tensor(precomp["ray_metric_ds_dz"].astype(np.float32), dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(ray_u, ray_v, indexing="xy")
    xx, yy = torch.meshgrid(x_centers, y_centers, indexing="xy")
    return WEDGeometryGPU(
        z_centers_np=z_centers_np,
        z_edges_np=z_edges_np,
        x_centers=x_centers,
        y_centers=y_centers,
        ray_u=ray_u,
        ray_v=ray_v,
        metric=metric,
        uu=uu,
        vv=vv,
        xx=xx,
        yy=yy,
    )


def build_inverse_square_master_gpu(precomp: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    z = torch.as_tensor(precomp["master_z_centers_mm"].astype(np.float32), dtype=torch.float32, device=device)[:, None, None]
    y = torch.as_tensor(precomp["cart_y_centers_mm"].astype(np.float32), dtype=torch.float32, device=device)[None, :, None]
    x = torch.as_tensor(precomp["cart_x_centers_mm"].astype(np.float32), dtype=torch.float32, device=device)[None, None, :]
    return (1000.0 / torch.sqrt(x * x + y * y + (z + 1000.0) * (z + 1000.0))).square()


def inverse_square_value(z_mm: float, y_mm: float, x_mm: float) -> float:
    return float((1000.0 / math.sqrt(x_mm * x_mm + y_mm * y_mm + (z_mm + 1000.0) * (z_mm + 1000.0))) ** 2)


def load_mlc_overlap_y_master_gpu(precomp: dict[str, np.ndarray], device: torch.device) -> torch.Tensor | None:
    if "mlc_overlap_y_master" not in precomp:
        return None
    return torch.as_tensor(precomp["mlc_overlap_y_master"], dtype=torch.float32, device=device)


def exact_mlc_area_fraction_shifted(
    cp: dict,
    machine,
    z_centers_mm: np.ndarray,
    y_centers_mm: np.ndarray,
    x_centers_mm: np.ndarray,
    device: torch.device,
    overlap_y_y_leaf: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    left = torch.as_tensor(cp["mlc_left_int_mm"], dtype=torch.float32, device=device)
    right = torch.as_tensor(cp["mlc_right_int_mm"], dtype=torch.float32, device=device)
    leaf_lo = -torch.maximum(left, right)
    leaf_hi = -torch.minimum(left, right)
    jaw_x0, jaw_x1 = [float(v) for v in machine.jaw_x_mm]
    jaw_y0, jaw_y1 = [float(v) for v in machine.jaw_y_mm]

    z = torch.as_tensor(z_centers_mm, dtype=torch.float32, device=device)
    scale = (float(machine.sad_mm) + z) / float(machine.sad_mm)
    x_left = torch.maximum(scale[:, None] * leaf_lo[None, :], scale[:, None] * jaw_x0)
    x_right = torch.minimum(scale[:, None] * leaf_hi[None, :], scale[:, None] * jaw_x1)

    spacing = 2.0
    x_edges = torch.as_tensor(np.r_[x_centers_mm[0] - spacing / 2.0, x_centers_mm + spacing / 2.0], dtype=torch.float32, device=device)
    y_edges = torch.as_tensor(np.r_[y_centers_mm[0] - spacing / 2.0, y_centers_mm + spacing / 2.0], dtype=torch.float32, device=device)
    pixel_x0, pixel_x1 = x_edges[:-1], x_edges[1:]

    overlap_x = (
        torch.minimum(x_right[:, :, None], pixel_x1[None, None, :])
        - torch.maximum(x_left[:, :, None], pixel_x0[None, None, :])
    ).clamp_min(0.0) / spacing
    if overlap_y_y_leaf is None:
        leaf_y_edges = torch.arange(machine.mlc_num_leaf_pairs + 1, dtype=torch.float32, device=device)
        leaf_y_edges = float(jaw_y0) + leaf_y_edges * float(machine.mlc_leaf_thickness_mm)
        y_bottom = torch.maximum(scale[:, None] * leaf_y_edges[:-1][None, :], scale[:, None] * jaw_y0)
        y_top = torch.minimum(scale[:, None] * leaf_y_edges[1:][None, :], scale[:, None] * jaw_y1)
        pixel_y0, pixel_y1 = y_edges[:-1], y_edges[1:]
        overlap_y_y_leaf_t = (
            torch.minimum(y_top[:, None, :], pixel_y1[None, :, None])
            - torch.maximum(y_bottom[:, None, :], pixel_y0[None, :, None])
        ).clamp_min(0.0) / spacing
    else:
        overlap_y_y_leaf_t = torch.as_tensor(overlap_y_y_leaf, dtype=torch.float32, device=device)
    return torch.bmm(overlap_y_y_leaf_t, overlap_x).clamp(0.0, 1.0)


def mlc_aperture_scalars(cp: dict, machine) -> tuple[float, float, float]:
    left = np.asarray(cp["mlc_left_int_mm"], dtype=np.float64)
    right = np.asarray(cp["mlc_right_int_mm"], dtype=np.float64)
    leaf_lo = -np.maximum(left, right)
    leaf_hi = -np.minimum(left, right)
    jaw_x0, jaw_x1 = machine.jaw_x_mm
    jaw_y0, _ = machine.jaw_y_mm
    leaf_y_edges = jaw_y0 + np.arange(machine.mlc_num_leaf_pairs + 1, dtype=np.float64) * machine.mlc_leaf_thickness_mm

    intervals: list[tuple[float, float, float, float]] = []
    for i in range(machine.mlc_num_leaf_pairs):
        x0 = max(float(leaf_lo[i]), float(jaw_x0))
        x1 = min(float(leaf_hi[i]), float(jaw_x1))
        y0 = float(leaf_y_edges[i])
        y1 = float(leaf_y_edges[i + 1])
        if x1 > x0 and y1 > y0:
            intervals.append((x0, x1, y0, y1))
    area = sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in intervals)
    perimeter = sum(2.0 * ((x1 - x0) + (y1 - y0)) for x0, x1, y0, y1 in intervals)
    for a, b in zip(intervals[:-1], intervals[1:]):
        if math.isclose(a[3], b[2], abs_tol=1e-5):
            shared = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
            perimeter -= 2.0 * shared
    s_equiv = 0.0 if area <= 0.0 or perimeter <= 0.0 else 4.0 * area / perimeter
    return float(area), float(perimeter), float(s_equiv)


def compute_wed(
    density_master: np.ndarray | torch.Tensor,
    output_start: int,
    output_end: int,
    wed_geometry: WEDGeometryGPU,
    rho_air: float,
    device: torch.device,
) -> torch.Tensor:
    z_centers = wed_geometry.z_centers_np
    cumulative = float(rho_air) * wed_geometry.metric * float(wed_geometry.z_edges_np[0] - (-1000.0))
    wed_out = torch.empty(
        (OUTPUT_DEPTH, wed_geometry.y_centers.numel(), wed_geometry.x_centers.numel()),
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        for z0 in range(0, output_end + 1, Z_CHUNK_SIZE):
            z1 = min(z0 + Z_CHUNK_SIZE, output_end + 1)
            if isinstance(density_master, torch.Tensor):
                density = density_master[z0:z1]
                if density.device != device or density.dtype != torch.float32:
                    density = density.to(device=device, dtype=torch.float32)
            else:
                density = torch.as_tensor(density_master[z0:z1], dtype=torch.float32, device=device)
            density_delta = density - float(rho_air)
            z_chunk = torch.as_tensor(z_centers[z0:z1], dtype=torch.float32, device=device)
            m = (1000.0 + z_chunk) / 1000.0

            sample_x = m[:, None, None] * wed_geometry.uu[None, :, :]
            sample_y = m[:, None, None] * wed_geometry.vv[None, :, :]
            grid = torch.stack(
                (
                    physical_to_grid(sample_x, wed_geometry.x_centers),
                    physical_to_grid(sample_y, wed_geometry.y_centers),
                ),
                dim=-1,
            )
            rho_ray = F.grid_sample(
                density_delta[:, None, :, :],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[:, 0] + float(rho_air)

            cell_wed = rho_ray * 2.0 * wed_geometry.metric[None, :, :]
            local_cumsum = torch.cumsum(cell_wed, dim=0)
            wed_ray = cumulative[None, :, :] + local_cumsum - 0.5 * cell_wed

            rel0 = max(output_start, z0)
            rel1 = min(output_end + 1, z1)
            if rel0 < rel1:
                local = slice(rel0 - z0, rel1 - z0)
                z_rel = torch.as_tensor(z_centers[rel0:rel1], dtype=torch.float32, device=device)
                denom = 1000.0 + z_rel
                u_target = wed_geometry.xx[None, :, :] * 1000.0 / denom[:, None, None]
                v_target = wed_geometry.yy[None, :, :] * 1000.0 / denom[:, None, None]
                valid = (torch.abs(u_target) <= 200.0) & (torch.abs(v_target) <= 200.0)
                out_grid = torch.stack(
                    (
                        physical_to_grid(u_target, wed_geometry.ray_u),
                        physical_to_grid(v_target, wed_geometry.ray_v),
                    ),
                    dim=-1,
                )
                wed_cart = F.grid_sample(
                    wed_ray[local, None, :, :],
                    out_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[:, 0]
                wed_cart = torch.where(valid, wed_cart, torch.zeros_like(wed_cart))
                wed_out[rel0 - output_start:rel1 - output_start] = wed_cart

            cumulative = cumulative + cell_wed.sum(dim=0)
    return wed_out


def write_nifti(path: Path, array: np.ndarray, geometry, z_centers: np.ndarray, y_centers: np.ndarray, x_centers: np.ndarray) -> None:
    sitk.WriteImage(make_shifted_bev_sitk(array, geometry, z_centers, y_centers, x_centers), str(path))


def write_maps_hdf5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    kwargs = {
        "compression": HDF5_COMPRESSION,
        "compression_opts": HDF5_COMPRESSION_OPTS,
        "shuffle": HDF5_SHUFFLE,
        "chunks": HDF5_CHUNKS,
    }
    if HDF5_COMPRESSION is None:
        kwargs = {"chunks": HDF5_CHUNKS}
    with h5py.File(path, "w") as handle:
        for name in OUTPUT_MAP_NAMES:
            handle.create_dataset(
                name,
                data=arrays[name].astype(np.float32, copy=False),
                dtype=np.float32,
                **kwargs,
            )
        handle.attrs["array_layout"] = "depth,superior_inferior,lateral"
        handle.attrs["shape_depth_si_lateral"] = (OUTPUT_DEPTH, 200, 200)
        handle.attrs["spacing_mm"] = 2.0


def sample_output_dir(task) -> Path:
    return OUTPUT_ROOT / task.group_name / task.key


def sample_is_complete(sample_dir: Path) -> bool:
    maps_path = sample_dir / HDF5_MAP_FILENAME
    if not (sample_dir / "metadata.json").exists() or not maps_path.exists():
        return False
    try:
        with h5py.File(maps_path, "r") as handle:
            return all(name in handle and handle[name].shape == (OUTPUT_DEPTH, 200, 200) for name in OUTPUT_MAP_NAMES)
    except OSError:
        return False


def select_task(tasks: list, key: str | None, index: int):
    if key:
        matches = [task for task in tasks if task.key == key]
        if not matches:
            raise ValueError(f"SAMPLE_KEY not found: {key}")
        return matches[0]
    return tasks[index]


def main() -> None:
    total_tic = time.perf_counter()
    precomp = load_precompute()
    training_root = PRE.photon_training_root(DATA_ROOT)
    machine = PRE.load_photon_machine_geometry(training_root)
    hu, density_curve = load_hu_density_curve(training_root)
    rho_air = float(density_curve[0])
    device = get_device()
    wed_geometry = build_wed_geometry_gpu(precomp, device)
    mlc_overlap_y_master_t = load_mlc_overlap_y_master_gpu(precomp, device)
    gpu_output = torch.empty((len(OUTPUT_MAP_NAMES), OUTPUT_DEPTH, 200, 200), dtype=torch.float32, device=device)
    host_output = torch.empty((len(OUTPUT_MAP_NAMES), OUTPUT_DEPTH, 200, 200), dtype=torch.float32, pin_memory=(device.type == "cuda"))

    selected_patient_ids = selected_patient_ids_from_range(training_root)
    tasks = PRE.discover_samples(training_root, selected_patient_ids, None, None, 0.2, None)
    task_index_by_key = {task.key: index for index, task in enumerate(tasks)}
    if PROCESS_ALL_SAMPLES:
        selected_tasks = tasks[:MAX_SAMPLES] if MAX_SAMPLES is not None else tasks
    else:
        selected_tasks = [select_task(tasks, SAMPLE_KEY, SAMPLE_INDEX)]

    z_master = precomp["master_z_centers_mm"].astype(np.float32)
    x_centers = precomp["cart_x_centers_mm"].astype(np.float32)
    y_centers = precomp["cart_y_centers_mm"].astype(np.float32)

    print(f"Samples selected: {len(selected_tasks)} / {len(tasks)}")
    print(f"Patient range: {PATIENT_ID_START}..{PATIENT_ID_END}")
    print(f"Output shape: ({OUTPUT_DEPTH}, 200, 200)")
    print(f"Device: {device}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Overwrite existing complete samples: {OVERWRITE_EXISTING}")

    current_patient_id: str | None = None
    patient_cache: PatientGpuCache | None = None
    processed = 0
    skipped_existing = 0

    progress = tqdm(selected_tasks, total=len(selected_tasks), desc="Preprocessing", unit="sample")
    for task in progress:
        sample_tic = time.perf_counter()
        task_index = task_index_by_key[task.key]
        sample_dir = sample_output_dir(task)
        progress.set_postfix_str(f"{task.key}")
        if sample_is_complete(sample_dir) and not OVERWRITE_EXISTING:
            skipped_existing += 1
            progress.set_postfix_str(f"skip existing {task.key}")
            continue

        geometry = PRE.geometry_from_beam_cp(task.beam, task.cp)

        patient_cache_reused = task.patient_id == current_patient_id and patient_cache is not None
        if not patient_cache_reused:
            if patient_cache is not None:
                del patient_cache
                patient_cache = None
            tqdm.write(f"Loading patient {task.patient_id}: {task.ct_path}")
            patient_cache = build_patient_gpu_cache(task, hu, density_curve, device)
            current_patient_id = task.patient_id
            tqdm.write(
                f"Cached patient {task.patient_id}: "
                f"{patient_cache.body_non_air_voxel_count} body voxels"
            )

        assert patient_cache is not None
        dose_img = PRE.read_float_image(task.dose_path)
        dose_array = sitk.GetArrayFromImage(dose_img).astype(np.float32, copy=False)
        dose_gpu = torch.as_tensor(dose_array, dtype=torch.float32, device=device)

        if USE_GPU_RESAMPLING:
            offset_info = compute_output_window_from_patient_cache(patient_cache, geometry, z_master, device)
        else:
            ct_array = patient_cache.ct_gpu.detach().cpu().numpy()
            ct_master, valid_master = resample_array_to_bev_coords(
                ct_array, patient_cache.ct_img, geometry, z_master, y_centers, x_centers,
                order=INTERP_ORDER, cval=CT_AIR_HU, chunk_size=Z_CHUNK_SIZE, return_valid=True,
            )
            occupied = np.any(valid_master & (ct_master > BODY_THRESHOLD_HU), axis=(1, 2))
            offset_info = compute_output_window(occupied, z_master)
        output_start = int(offset_info["output_start_index"])
        output_end = int(offset_info["output_end_index"])
        z_output = z_master[output_start:output_end + 1]

        if not USE_GPU_RESAMPLING:
            raise RuntimeError("CPU fallback is not maintained for the optimized patient-cache path.")

        ct_master_grid = build_bev_sampling_grid_gpu(
            patient_cache.ct_img, geometry, z_master[:output_end + 1], y_centers, x_centers, device=device
        )
        output_grid_ct = ct_master_grid[:, output_start:output_end + 1]
        ct_bev_t = sample_with_prebuilt_grid(
            patient_cache.ct_gpu[None, None], output_grid_ct, cval=CT_AIR_HU
        ).clamp_(CT_AIR_HU, 3071.0)
        density_master_needed_t = sample_with_prebuilt_grid(
            patient_cache.density_gpu[None, None], ct_master_grid, cval=rho_air
        )
        density_bev_t = density_master_needed_t[output_start:output_end + 1]
        output_grid_mr = output_grid_ct if PRE.same_geometry(patient_cache.mr_ref_img, patient_cache.ct_img) else build_bev_sampling_grid_gpu(
            patient_cache.mr_ref_img, geometry, z_output, y_centers, x_centers, device=device
        )
        output_grid_dose = output_grid_ct if PRE.same_geometry(dose_img, patient_cache.ct_img) else build_bev_sampling_grid_gpu(
            dose_img, geometry, z_output, y_centers, x_centers, device=device
        )
        mr_bev_t = sample_with_prebuilt_grid(patient_cache.mr_gpu[None, None], output_grid_mr, cval=0.0)
        dose_bev_t = sample_with_prebuilt_grid(dose_gpu[None, None], output_grid_dose, cval=0.0).clamp_min_(0.0)
        wed_bev_t = compute_wed(density_master_needed_t, output_start, output_end, wed_geometry, rho_air, device)

        mlc_overlap_y = None
        if mlc_overlap_y_master_t is not None:
            mlc_overlap_y = mlc_overlap_y_master_t[output_start:output_end + 1]
        mlc_fraction_t = exact_mlc_area_fraction_shifted(
            task.cp,
            machine,
            z_output,
            y_centers,
            x_centers,
            device,
            overlap_y_y_leaf=mlc_overlap_y,
        )

        map_names = list(OUTPUT_MAP_NAMES)
        gpu_output[0].copy_(ct_bev_t)
        gpu_output[1].copy_(mr_bev_t)
        gpu_output[2].copy_(density_bev_t)
        gpu_output[3].copy_(wed_bev_t)
        gpu_output[4].copy_(mlc_fraction_t)
        gpu_output[5].copy_(dose_bev_t)
        host_output.copy_(gpu_output, non_blocking=(device.type == "cuda"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        arrays = {name: host_output[index].numpy() for index, name in enumerate(map_names)}

        for name, array in arrays.items():
            if array.shape != (OUTPUT_DEPTH, 200, 200):
                raise RuntimeError(f"{name} has wrong shape {array.shape}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"{name} contains NaN/Inf")
        mlc_fraction = arrays["proj_fraction"]
        wed_bev = arrays["wed"]
        density_bev = arrays["mass_density"]
        if not (-1e-5 <= float(mlc_fraction.min()) <= float(mlc_fraction.max()) <= 1.0 + 1e-5):
            raise RuntimeError("MLC fractional projection is outside [0,1]")
        if abs(offset_info["bev_z_offset_mm"] / 2.0 - round(offset_info["bev_z_offset_mm"] / 2.0)) > 1e-6:
            raise RuntimeError("bev_z_offset_mm is not a 2 mm integer multiple")

        a_open, p_open, s_equiv = mlc_aperture_scalars(task.cp, machine)
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_maps_hdf5(sample_dir / HDF5_MAP_FILENAME, arrays)

        metadata = {
            "key": task.key,
            "sample_index": task_index,
            "group": task.group_name,
            "patient_id": task.patient_id,
            "beam_index": task.beam_index,
            "cp_index": task.cp_index,
            "ct_path": str(patient_cache.ct_path),
            "mr_path": str(patient_cache.mr_path) if patient_cache.mr_path is not None else None,
            "dose_path": str(task.dose_path),
            "shape_depth_si_lateral": [OUTPUT_DEPTH, 200, 200],
            "spacing_mm": 2.0,
            "A_open_mm2": a_open,
            "P_open_mm": p_open,
            "S_equivalent_mm": s_equiv,
            **offset_info,
            "density_unit": "g/cm3",
            "wed_unit": "mm_water",
            "rho_air_g_cm3": rho_air,
            "hu_to_density_source": str(training_root / "beam_parameters.json"),
            "geometry": asdict(geometry),
            "files": {
                "maps_hdf5": HDF5_MAP_FILENAME,
                "datasets": {name: name for name in arrays},
            },
            "hdf5": {
                "array_layout": "depth,superior_inferior,lateral",
                "compression": HDF5_COMPRESSION,
                "compression_opts": HDF5_COMPRESSION_OPTS,
                "shuffle": HDF5_SHUFFLE,
                "chunks": list(HDF5_CHUNKS),
            },
            "elapsed_s": time.perf_counter() - sample_tic,
            "patient_cache_reused": patient_cache_reused,
            "qc": {
                "all_shapes": {name: list(array.shape) for name, array in arrays.items()},
                "proj_fraction_min": float(mlc_fraction.min()),
                "proj_fraction_max": float(mlc_fraction.max()),
                "inverse_square_nearest_isocenter": inverse_square_value(
                    float(z_output[np.argmin(np.abs(z_output))]),
                    float(y_centers[99]),
                    float(x_centers[99]),
                ),
                "wed_min": float(wed_bev.min()),
                "wed_max": float(wed_bev.max()),
                "mass_density_min": float(density_bev.min()),
                "mass_density_max": float(density_bev.max()),
            },
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        processed += 1
        progress.set_postfix_str(
            f"wrote {task.key}, offset={offset_info['bev_z_offset_mm']:.0f}mm, "
            f"elapsed={metadata['elapsed_s']:.2f}s"
        )

    print(
        f"\nDone. Processed {processed} sample(s), skipped existing {skipped_existing}, "
        f"elapsed {time.perf_counter() - total_tic:.2f} s"
    )


if __name__ == "__main__":
    main()
