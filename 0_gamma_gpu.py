#!/usr/bin/env python3
"""
GPU-accelerated 3D global gamma pass-rate calculation.

This module is intentionally focused on pass rate, not on exporting a full
gamma volume. It evaluates only the foreground reference voxels and searches
sub-voxel evaluation points with trilinear interpolation via torch.grid_sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GammaGPUConfig:
    spacing_mm: float = 2.0
    interp_fraction: float = 5.0
    point_chunk_size: int = 32768
    offset_chunk_size: int = 128
    skip_once_passed: bool = True
    device: str = "cuda:0"


def make_search_offsets(
    distance_mm: float,
    spacing_mm: float,
    interp_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sub-voxel offsets in z/y/x voxel units and their mm distances."""
    if distance_mm <= 0:
        raise ValueError("distance_mm must be positive.")
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be positive.")
    if interp_fraction <= 0:
        raise ValueError("interp_fraction must be positive.")

    step_mm = float(distance_mm) / float(interp_fraction)
    steps = int(np.ceil(float(distance_mm) / step_mm))
    coords_mm = torch.arange(-steps, steps + 1, device=device, dtype=torch.float32) * step_mm
    zz, yy, xx = torch.meshgrid(coords_mm, coords_mm, coords_mm, indexing="ij")
    distances_mm = torch.sqrt(zz * zz + yy * yy + xx * xx)
    keep = distances_mm <= (float(distance_mm) + 1e-6)

    offsets_mm = torch.stack((zz[keep], yy[keep], xx[keep]), dim=1)
    distances_mm = distances_mm[keep]
    order = torch.argsort(distances_mm)
    offsets_voxels = offsets_mm[order] / float(spacing_mm)
    return offsets_voxels.contiguous(), distances_mm[order].contiguous()


def _normalise_grid_coordinates(
    zyx_coords: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Convert z/y/x voxel-index coordinates to grid_sample x/y/z coordinates."""
    depth, height, width = shape
    z = zyx_coords[:, :, 0]
    y = zyx_coords[:, :, 1]
    x = zyx_coords[:, :, 2]

    x_norm = 2.0 * x / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y / max(height - 1, 1) - 1.0
    z_norm = 2.0 * z / max(depth - 1, 1) - 1.0
    return torch.stack((x_norm, y_norm, z_norm), dim=-1)


def gamma_pass_rate_gpu(
    prediction: np.ndarray | torch.Tensor,
    reference: np.ndarray | torch.Tensor,
    foreground: np.ndarray | torch.Tensor,
    dose_percent_threshold: float,
    distance_mm_threshold: float,
    reference_max: float | None = None,
    config: GammaGPUConfig | None = None,
) -> float:
    """
    Calculate a 3D global gamma pass rate on GPU.

    Parameters
    ----------
    prediction, reference:
        Dose grids with shape [Z, Y, X], in the same scaled dose units.
    foreground:
        Boolean mask with shape [Z, Y, X]. Gamma is calculated and counted only
        at these reference voxel centres.
    dose_percent_threshold:
        Global dose difference criterion in percent of reference_max.
    distance_mm_threshold:
        Distance-to-agreement criterion in mm.
    reference_max:
        Global normalisation dose. If None, max(reference) is used.
    config:
        Runtime and interpolation settings.
    """
    config = config or GammaGPUConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    pred = torch.as_tensor(prediction, dtype=torch.float32, device=device).contiguous()
    ref = torch.as_tensor(reference, dtype=torch.float32, device=device).contiguous()
    fg = torch.as_tensor(foreground, dtype=torch.bool, device=device)

    if pred.shape != ref.shape or pred.shape != fg.shape:
        raise ValueError(f"prediction, reference, and foreground must have the same shape; got {pred.shape}, {ref.shape}, {fg.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D [Z,Y,X] arrays, got shape {pred.shape}")
    if not torch.any(fg):
        return float("nan")

    if reference_max is None:
        reference_max_value = float(torch.max(ref).detach().cpu())
    else:
        reference_max_value = float(reference_max)
    dose_criterion = reference_max_value * float(dose_percent_threshold) / 100.0
    if dose_criterion <= 0:
        return float("nan")

    offsets, distances = make_search_offsets(
        float(distance_mm_threshold),
        float(config.spacing_mm),
        float(config.interp_fraction),
        device,
    )
    distance_term_sq = (distances / float(distance_mm_threshold)) ** 2

    eval_volume = pred.view(1, 1, *pred.shape)
    foreground_indices = torch.nonzero(fg, as_tuple=False).to(torch.float32)
    total_foreground = int(foreground_indices.shape[0])
    passed = 0

    for point_start in range(0, total_foreground, int(config.point_chunk_size)):
        point_end = min(point_start + int(config.point_chunk_size), total_foreground)
        point_coords = foreground_indices[point_start:point_end]
        ref_values = ref[
            point_coords[:, 0].long(),
            point_coords[:, 1].long(),
            point_coords[:, 2].long(),
        ]

        min_gamma_sq = torch.full((point_coords.shape[0],), torch.inf, dtype=torch.float32, device=device)
        active = torch.arange(point_coords.shape[0], device=device)

        for offset_start in range(0, offsets.shape[0], int(config.offset_chunk_size)):
            if active.numel() == 0:
                break

            offset_end = min(offset_start + int(config.offset_chunk_size), offsets.shape[0])
            current_offsets = offsets[offset_start:offset_end]
            current_distance_term_sq = distance_term_sq[offset_start:offset_end]

            active_coords = point_coords[active]
            query_zyx = active_coords[:, None, :] + current_offsets[None, :, :]
            depth, height, width = pred.shape
            valid = (
                (query_zyx[:, :, 0] >= 0)
                & (query_zyx[:, :, 0] <= depth - 1)
                & (query_zyx[:, :, 1] >= 0)
                & (query_zyx[:, :, 1] <= height - 1)
                & (query_zyx[:, :, 2] >= 0)
                & (query_zyx[:, :, 2] <= width - 1)
            )
            if not torch.any(valid):
                continue

            grid = _normalise_grid_coordinates(query_zyx, tuple(int(v) for v in pred.shape))
            grid = grid.view(1, active_coords.shape[0], current_offsets.shape[0], 1, 3)
            sampled = F.grid_sample(
                eval_volume,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(active_coords.shape[0], current_offsets.shape[0])

            dose_term_sq = ((sampled - ref_values[active, None]) / dose_criterion) ** 2
            gamma_sq = dose_term_sq + current_distance_term_sq[None, :]
            gamma_sq = torch.where(valid, gamma_sq, torch.full_like(gamma_sq, torch.inf))
            min_gamma_sq[active] = torch.minimum(min_gamma_sq[active], torch.min(gamma_sq, dim=1).values)

            if config.skip_once_passed:
                active = active[min_gamma_sq[active] > 1.0]

        passed += int(torch.count_nonzero(min_gamma_sq <= 1.0).detach().cpu())

    return passed / float(total_foreground)


def _demo() -> None:
    shape = (32, 48, 48)
    ref = np.zeros(shape, dtype=np.float32)
    ref[8:24, 12:36, 12:36] = 1.0
    pred = ref.copy()
    foreground = ref >= 0.1 * ref.max()
    config = GammaGPUConfig(interp_fraction=5, device="cuda:0")
    for dose_percent, distance_mm in ((1, 1), (2, 2), (3, 3)):
        value = gamma_pass_rate_gpu(
            pred,
            ref,
            foreground,
            dose_percent_threshold=dose_percent,
            distance_mm_threshold=distance_mm,
            reference_max=float(ref.max()),
            config=config,
        )
        print(f"{distance_mm:g}mm/{dose_percent:g}% gamma pass rate = {value:.6f}")


if __name__ == "__main__":
    _demo()
