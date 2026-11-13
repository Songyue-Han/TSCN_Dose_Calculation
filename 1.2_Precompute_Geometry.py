#!/usr/bin/env python3
"""
Precompute DoseRAD2026 BEV geometry shared by all CT->dose samples.

The per-sample preprocessing in 1.3_New_Preprocessing.py slices these master
arrays according to each sample's dynamic z offset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


OUTPUT_DIR = Path(__file__).resolve().parent / "precomputed_geometry_v1_2"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "DoseRAD2026"

SPACING_MM = 2.0
SAD_MM = 1000.0
SOURCE_Z_MM = -1000.0

OUTPUT_SHAPE_DSL = (320, 200, 200)
EXTRA_UPSTREAM_SLICES = 250
EXTRA_DOWNSTREAM_SLICES = 250
MASTER_DEPTH = OUTPUT_SHAPE_DSL[0] + EXTRA_UPSTREAM_SLICES + EXTRA_DOWNSTREAM_SLICES

RAY_UV_MIN_MM = -200.0
RAY_UV_MAX_MM = 200.0
RAY_UV_SPACING_MM = 1.0


def photon_training_root(data_root: Path) -> Path:
    candidates = [
        data_root / "photon" / "training",
        data_root / "DoseRAD2026" / "photon" / "training",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find photon/training under {data_root}")


def load_photon_machine_geometry(data_root: Path) -> dict:
    with open(photon_training_root(data_root) / "beam_parameters.json", "r") as f:
        return json.load(f)["photon"]


def even_grid_centers(num_voxels: int, spacing_mm: float) -> np.ndarray:
    return (np.arange(num_voxels, dtype=np.float32) - (num_voxels - 1) / 2.0) * spacing_mm


def even_grid_edges(num_voxels: int, spacing_mm: float) -> np.ndarray:
    return (np.arange(num_voxels + 1, dtype=np.float32) - num_voxels / 2.0) * spacing_mm


def write_bev_identity_nifti(path: Path, array_dyx: np.ndarray, spacing_mm: float) -> None:
    image = sitk.GetImageFromArray(array_dyx.astype(np.float32))
    image.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    image.SetOrigin(
        (
            float(even_grid_centers(array_dyx.shape[2], spacing_mm)[0]),
            float(even_grid_centers(array_dyx.shape[1], spacing_mm)[0]),
            float(even_grid_centers(array_dyx.shape[0], spacing_mm)[0]),
        )
    )
    image.SetDirection(tuple(np.eye(3, dtype=np.float64).reshape(-1)))
    sitk.WriteImage(image, str(path))


def precompute_mlc_overlap_y_master(
    master_z_centers_mm: np.ndarray,
    cart_y_centers_mm: np.ndarray,
    machine: dict,
) -> np.ndarray:
    mlc_num_leaf_pairs = int(machine["mlc_num_leaf_pairs"])
    leaf_thickness_mm = float(machine["mlc_leaf_thickness_mm"])
    jaw_y0, jaw_y1 = [float(v) for v in machine["jaw_y_mm"]]

    scale = ((SAD_MM + master_z_centers_mm) / SAD_MM).astype(np.float32)
    leaf_y_edges = jaw_y0 + np.arange(mlc_num_leaf_pairs + 1, dtype=np.float32) * leaf_thickness_mm
    y_edges = np.concatenate(
        (
            np.asarray([cart_y_centers_mm[0] - SPACING_MM / 2.0], dtype=np.float32),
            cart_y_centers_mm + SPACING_MM / 2.0,
        )
    )
    pixel_y0 = y_edges[:-1]
    pixel_y1 = y_edges[1:]

    y_bottom = np.maximum(scale[:, None] * leaf_y_edges[:-1][None, :], scale[:, None] * jaw_y0)
    y_top = np.minimum(scale[:, None] * leaf_y_edges[1:][None, :], scale[:, None] * jaw_y1)
    overlap_y = (
        np.minimum(y_top[:, None, :], pixel_y1[None, :, None])
        - np.maximum(y_bottom[:, None, :], pixel_y0[None, :, None])
    )
    return np.clip(overlap_y / SPACING_MM, 0.0, 1.0).astype(np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    machine = load_photon_machine_geometry(DEFAULT_DATA_ROOT)

    master_z_centers_mm = even_grid_centers(MASTER_DEPTH, SPACING_MM)
    master_z_edges_mm = even_grid_edges(MASTER_DEPTH, SPACING_MM)
    cart_x_centers_mm = even_grid_centers(OUTPUT_SHAPE_DSL[2], SPACING_MM)
    cart_y_centers_mm = even_grid_centers(OUTPUT_SHAPE_DSL[1], SPACING_MM)
    ray_u_mm = np.arange(RAY_UV_MIN_MM, RAY_UV_MAX_MM + 0.5 * RAY_UV_SPACING_MM, RAY_UV_SPACING_MM, dtype=np.float32)
    ray_v_mm = np.arange(RAY_UV_MIN_MM, RAY_UV_MAX_MM + 0.5 * RAY_UV_SPACING_MM, RAY_UV_SPACING_MM, dtype=np.float32)

    z = master_z_centers_mm[:, None, None]
    y = cart_y_centers_mm[None, :, None]
    x = cart_x_centers_mm[None, None, :]
    r_mm = np.sqrt(x * x + y * y + (z - SOURCE_Z_MM) * (z - SOURCE_Z_MM), dtype=np.float32)
    master_inverse_square = np.square(SAD_MM / r_mm, dtype=np.float32)

    u = ray_u_mm[None, :]
    v = ray_v_mm[:, None]
    ray_metric_ds_dz = np.sqrt(1.0 + (u / SAD_MM) ** 2 + (v / SAD_MM) ** 2, dtype=np.float32)
    magnification_by_z = (SAD_MM + master_z_centers_mm) / SAD_MM
    mlc_overlap_y_master = precompute_mlc_overlap_y_master(master_z_centers_mm, cart_y_centers_mm, machine)

    npz_path = OUTPUT_DIR / "precompute_geometry.npz"
    np.savez_compressed(
        npz_path,
        spacing_mm=np.float32(SPACING_MM),
        sad_mm=np.float32(SAD_MM),
        source_z_mm=np.float32(SOURCE_Z_MM),
        output_shape_dsl=np.asarray(OUTPUT_SHAPE_DSL, dtype=np.int32),
        master_shape_dsl=np.asarray((MASTER_DEPTH, OUTPUT_SHAPE_DSL[1], OUTPUT_SHAPE_DSL[2]), dtype=np.int32),
        extra_upstream_slices=np.int32(EXTRA_UPSTREAM_SLICES),
        extra_downstream_slices=np.int32(EXTRA_DOWNSTREAM_SLICES),
        master_z_centers_mm=master_z_centers_mm,
        master_z_edges_mm=master_z_edges_mm,
        cart_x_centers_mm=cart_x_centers_mm,
        cart_y_centers_mm=cart_y_centers_mm,
        ray_u_mm=ray_u_mm,
        ray_v_mm=ray_v_mm,
        magnification_by_z=magnification_by_z.astype(np.float32),
        ray_metric_ds_dz=ray_metric_ds_dz.astype(np.float32),
        mlc_overlap_y_master=mlc_overlap_y_master,
    )

    inverse_square_path = OUTPUT_DIR / "master_inverse_square.nii.gz"
    write_bev_identity_nifti(inverse_square_path, master_inverse_square, SPACING_MM)

    metadata = {
        "description": "Shared BEV geometry for DoseRAD2026 1.3 preprocessing.",
        "spacing_mm": SPACING_MM,
        "sad_mm": SAD_MM,
        "source_xyz_bev_mm": [0.0, 0.0, SOURCE_Z_MM],
        "output_shape_depth_si_lateral": list(OUTPUT_SHAPE_DSL),
        "master_shape_depth_si_lateral": [MASTER_DEPTH, OUTPUT_SHAPE_DSL[1], OUTPUT_SHAPE_DSL[2]],
        "default_output_start_index": EXTRA_UPSTREAM_SLICES,
        "allowed_offset_slices": EXTRA_UPSTREAM_SLICES,
        "allowed_offset_mm": EXTRA_UPSTREAM_SLICES * SPACING_MM,
        "cartesian_grid_note": "Even grids use centers -199,-197,...,+199 mm for x/y and -319..+319 mm for the default output z window.",
        "ray_grid_note": "u,v are isocenter-plane ray coordinates in mm, sampled at 1 mm over [-200,+200].",
        "mlc_overlap_y_master_shape": list(mlc_overlap_y_master.shape),
        "mlc_overlap_y_master_note": "Shape [master_z, cart_y, leaf_pair]. Precomputed exact fractional y-overlap between divergent leaf strips and BEV y pixels; independent of control-point leaf openings.",
        "files": {
            "geometry_npz": npz_path.name,
            "master_inverse_square_nifti": inverse_square_path.name,
        },
    }
    with open(OUTPUT_DIR / "precompute_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {npz_path}")
    print(f"Wrote {inverse_square_path}")
    print(f"master_inverse_square shape: {master_inverse_square.shape}")
    print(f"mlc_overlap_y_master shape: {mlc_overlap_y_master.shape}")
    print(f"iso-nearest inverse-square: {master_inverse_square[409, 99, 99]:.6f}")


if __name__ == "__main__":
    main()
