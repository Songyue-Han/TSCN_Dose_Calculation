#!/usr/bin/env python3
"""
BEV preprocessing for DoseRAD2026 photon data.

It resamples:
  - CT into a beam-aligned cuboid
  - the matching segment dose into the same cuboid
  - the MLC aperture into a divergent BEV projection mask

For this first BEV version, the MLC aperture is treated as being defined on the
isocentre plane, then projected from the virtual photon source through each BEV
voxel back to that plane.

Output tensor convention:
  HDF5 datasets are shaped (depth, superior_inferior, lateral) = (256, 200, 200)
  at 2 mm isotropic spacing.

Output layout:
  1_Preprocessing/
    train/
      1ABB006_B0_CP000/
        ct.h5
        proj.h5
        dose.h5
        metadata.json
    validation/
      ...
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import SimpleITK as sitk


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "DoseRAD2026"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "1_Preprocessing"
DEFAULT_SHAPE_DSL = (256, 200, 200)  # depth, superior-inferior, lateral
DEFAULT_SPACING_MM = 2.0


@dataclass(frozen=True)
class SampleTask:
    patient_id: str
    patient_dir: Path
    group_name: str
    beam: dict
    cp: dict
    ct_path: Path
    dose_path: Path

    @property
    def beam_index(self) -> int:
        return int(self.beam["beam_idx"])

    @property
    def cp_index(self) -> int:
        return int(self.cp["cp_idx"])

    @property
    def key(self) -> str:
        return f"{self.patient_id}_B{self.beam_index}_CP{self.cp_index:03d}"


@dataclass(frozen=True)
class BeamGeometry:
    iso_center_xyz: Tuple[float, float, float]
    gantry_angle_deg: float
    beam_axis_xyz: Tuple[float, float, float]
    lateral_axis_xyz: Tuple[float, float, float]
    si_axis_xyz: Tuple[float, float, float]


@dataclass(frozen=True)
class PhotonMachineGeometry:
    sad_mm: float
    jaw_x_mm: Tuple[float, float]
    jaw_y_mm: Tuple[float, float]
    mlc_num_leaf_pairs: int
    mlc_leaf_thickness_mm: float


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        return path.read_text(errors="ignore").startswith("version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def assert_real_image(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if is_lfs_pointer(path):
        raise RuntimeError(f"{path} is still a Git LFS pointer, not a real image.")


def photon_training_root(data_root: Path) -> Path:
    candidates = [
        data_root / "photon" / "training",
        data_root / "DoseRAD2026" / "photon" / "training",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find photon/training under {data_root}")


def patient_dirs(training_root: Path) -> list[Path]:
    dirs = sorted(
        p for p in training_root.iterdir()
        if p.is_dir() and (p / f"{p.name}.json").exists()
    )
    if not dirs:
        raise RuntimeError(f"No patient folders with patient JSON files found in {training_root}")
    return dirs


def load_patient_plan(patient_dir: Path) -> dict:
    plan_path = patient_dir / f"{patient_dir.name}.json"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    with open(plan_path, "r") as f:
        return json.load(f)


def load_photon_machine_geometry(training_root: Path) -> PhotonMachineGeometry:
    params_path = training_root / "beam_parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    with open(params_path, "r") as f:
        params = json.load(f)["photon"]
    return PhotonMachineGeometry(
        sad_mm=float(params["SAD_mm"]),
        jaw_x_mm=tuple(float(v) for v in params["jaw_x_mm"]),
        jaw_y_mm=tuple(float(v) for v in params["jaw_y_mm"]),
        mlc_num_leaf_pairs=int(params["mlc_num_leaf_pairs"]),
        mlc_leaf_thickness_mm=float(params["mlc_leaf_thickness_mm"]),
    )


def split_patient_dirs(
    dirs: list[Path],
    validation_fraction: float,
    validation_patient_ids: set[str] | None,
) -> dict[str, str]:
    if validation_patient_ids is not None:
        unknown = sorted(validation_patient_ids - {p.name for p in dirs})
        if unknown:
            raise ValueError(f"Validation patient ids not found: {', '.join(unknown)}")
        return {
            patient_dir.name: "validation" if patient_dir.name in validation_patient_ids else "train"
            for patient_dir in dirs
        }

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be >= 0 and < 1")
    n_validation = int(round(len(dirs) * validation_fraction))
    validation_ids = {p.name for p in dirs[-n_validation:]} if n_validation else set()
    return {
        patient_dir.name: "validation" if patient_dir.name in validation_ids else "train"
        for patient_dir in dirs
    }


def discover_samples(
    training_root: Path,
    selected_patient_ids: set[str] | None,
    selected_beam_index: int | None,
    selected_cp_index: int | None,
    validation_fraction: float,
    validation_patient_ids: set[str] | None,
) -> list[SampleTask]:
    dirs = patient_dirs(training_root)
    if selected_patient_ids is not None:
        missing = sorted(selected_patient_ids - {p.name for p in dirs})
        if missing:
            raise ValueError(f"Selected patient ids not found: {', '.join(missing)}")
        dirs = [p for p in dirs if p.name in selected_patient_ids]

    group_by_patient = split_patient_dirs(dirs, validation_fraction, validation_patient_ids)
    tasks: list[SampleTask] = []

    for patient_dir in dirs:
        ct_path = patient_dir / "image" / "ct.mha"
        assert_real_image(ct_path)
        plan = load_patient_plan(patient_dir)
        group_name = group_by_patient[patient_dir.name]
        for beam in sorted(plan.get("beams", []), key=lambda item: int(item["beam_idx"])):
            beam_index = int(beam["beam_idx"])
            if selected_beam_index is not None and beam_index != selected_beam_index:
                continue

            for cp in sorted(beam.get("control_points", []), key=lambda item: int(item["cp_idx"])):
                cp_index = int(cp["cp_idx"])
                if selected_cp_index is not None and cp_index != selected_cp_index:
                    continue
                dose_path = patient_dir / "dose" / f"Dose_B{beam_index}_CP{cp_index:03d}.mha"
                assert_real_image(dose_path)
                tasks.append(
                    SampleTask(
                        patient_id=patient_dir.name,
                        patient_dir=patient_dir,
                        group_name=group_name,
                        beam=beam,
                        cp=cp,
                        ct_path=ct_path,
                        dose_path=dose_path,
                    )
                )

    return tasks


def limit_samples(
    tasks: list[SampleTask],
    max_patients: int | None,
    max_samples: int | None,
) -> list[SampleTask]:
    limited = tasks
    if max_patients is not None:
        allowed: set[str] = set()
        for task in limited:
            allowed.add(task.patient_id)
            if len(allowed) >= max_patients:
                break
        limited = [task for task in limited if task.patient_id in allowed]
    if max_samples is not None:
        limited = limited[:max_samples]
    return limited


def resume_tasks(
    tasks: list[SampleTask],
    resume_from_index: int | None,
    resume_from_key: str | None,
) -> tuple[list[SampleTask], int, str | None]:
    if resume_from_index is None and resume_from_key is None:
        return tasks, 0, None

    start_index: int | None = None
    messages: list[str] = []

    if resume_from_index is not None:
        if resume_from_index < 1 or resume_from_index > len(tasks):
            raise ValueError(
                f"--resume-from-index must be between 1 and {len(tasks)}, got {resume_from_index}"
            )
        start_index = resume_from_index - 1
        messages.append(f"index {resume_from_index}/{len(tasks)} -> {tasks[start_index].key}")

    if resume_from_key is not None:
        matching = [idx for idx, task in enumerate(tasks) if task.key == resume_from_key]
        if not matching:
            raise ValueError(f"--resume-from-key was not found in discovered tasks: {resume_from_key}")
        key_index = matching[0]
        messages.append(f"key {resume_from_key} -> index {key_index + 1}/{len(tasks)}")
        if start_index is not None and key_index != start_index:
            raise ValueError(
                "--resume-from-index and --resume-from-key point to different samples: "
                f"index gives {tasks[start_index].key}, key is at index {key_index + 1}"
            )
        start_index = key_index

    assert start_index is not None
    return tasks[start_index:], start_index, "; ".join(messages)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def geometry_from_beam_cp(beam: dict, cp: dict) -> BeamGeometry:
    iso = np.asarray(beam["iso_center"], dtype=np.float64)
    gantry_deg = float(cp["gantry_angle"])
    theta = math.radians(gantry_deg)

    # DoseRAD photon gantry angles follow the radiotherapy AP/PA convention in
    # the patient XY plane: gantry 0 deg is AP (anterior -> posterior, +Y in
    # these LPS-like images), and gantry 180 deg is PA (posterior -> anterior,
    # -Y). This is the ray direction from source toward isocenter.
    beam_axis = normalize(np.asarray([-math.sin(theta), math.cos(theta), 0.0], dtype=np.float64))
    si_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    lateral_axis = normalize(np.cross(si_axis, beam_axis))

    return BeamGeometry(
        iso_center_xyz=tuple(float(v) for v in iso),
        gantry_angle_deg=gantry_deg,
        beam_axis_xyz=tuple(float(v) for v in beam_axis),
        lateral_axis_xyz=tuple(float(v) for v in lateral_axis),
        si_axis_xyz=tuple(float(v) for v in si_axis),
    )


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def read_float_image(path: Path) -> sitk.Image:
    assert_real_image(path)
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def resample_dose_to_ct_if_needed(dose_img: sitk.Image, ct_img: sitk.Image) -> sitk.Image:
    if same_geometry(dose_img, ct_img):
        return dose_img
    return sitk.Resample(
        dose_img,
        ct_img,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def build_bev_grid_world(
    geometry: BeamGeometry,
    shape_dsl: Tuple[int, int, int],
    spacing_mm: float,
) -> np.ndarray:
    depth_n, si_n, lateral_n = shape_dsl
    depth_axis = np.asarray(geometry.beam_axis_xyz, dtype=np.float64)
    si_axis = np.asarray(geometry.si_axis_xyz, dtype=np.float64)
    lateral_axis = np.asarray(geometry.lateral_axis_xyz, dtype=np.float64)
    iso = np.asarray(geometry.iso_center_xyz, dtype=np.float64)

    depth = (np.arange(depth_n, dtype=np.float64) - (depth_n - 1) / 2.0) * spacing_mm
    si = (np.arange(si_n, dtype=np.float64) - (si_n - 1) / 2.0) * spacing_mm
    lateral = (np.arange(lateral_n, dtype=np.float64) - (lateral_n - 1) / 2.0) * spacing_mm

    d_grid, si_grid, lat_grid = np.meshgrid(depth, si, lateral, indexing="ij")
    world = (
        iso
        + d_grid[..., None] * depth_axis
        + si_grid[..., None] * si_axis
        + lat_grid[..., None] * lateral_axis
    )
    return world


def world_to_continuous_indices(image: sitk.Image, world_xyz: np.ndarray) -> np.ndarray:
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    flat_world = world_xyz.reshape(-1, 3)
    continuous_xyz = ((flat_world - origin) @ direction) / spacing

    # scipy/map_coordinates array order is z, y, x, while SimpleITK physical
    # points and image indices are x, y, z.
    return continuous_xyz[:, ::-1].T


def map_coordinates_cpu(
    array_zyx: np.ndarray,
    coords_zyx: np.ndarray,
    order: int,
    cval: float,
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    sampled = map_coordinates(
        array_zyx,
        coords_zyx,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=(order > 1),
    )
    return sampled.reshape(output_shape).astype(np.float32)


def resample_to_bev(
    image: sitk.Image,
    geometry: BeamGeometry,
    shape_dsl: Tuple[int, int, int],
    spacing_mm: float,
    order: int,
    cval: float,
) -> np.ndarray:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    return resample_array_to_bev(array, image, geometry, shape_dsl, spacing_mm, order, cval)


def resample_array_to_bev(
    array_zyx: np.ndarray,
    image: sitk.Image,
    geometry: BeamGeometry,
    shape_dsl: Tuple[int, int, int],
    spacing_mm: float,
    order: int,
    cval: float,
) -> np.ndarray:
    world = build_bev_grid_world(geometry, shape_dsl, spacing_mm)
    coords = world_to_continuous_indices(image, world)
    return map_coordinates_cpu(array_zyx, coords, order=order, cval=cval, output_shape=shape_dsl)


def build_mlc_aperture_isoplane(
    cp: dict,
    machine: PhotonMachineGeometry,
    resolution_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a 2D isoplane aperture mask arranged as (SI, lateral)."""
    x_min, x_max = machine.jaw_x_mm
    y_min, y_max = machine.jaw_y_mm
    lateral = np.arange(x_min + resolution_mm / 2.0, x_max, resolution_mm, dtype=np.float32)
    si = np.arange(y_min + resolution_mm / 2.0, y_max, resolution_mm, dtype=np.float32)
    mask = np.zeros((si.size, lateral.size), dtype=np.float32)

    left = np.asarray(cp["mlc_left_int_mm"], dtype=np.float32)
    right = np.asarray(cp["mlc_right_int_mm"], dtype=np.float32)
    if left.size != machine.mlc_num_leaf_pairs or right.size != machine.mlc_num_leaf_pairs:
        raise ValueError(
            f"Expected {machine.mlc_num_leaf_pairs} MLC leaf pairs, "
            f"got left={left.size}, right={right.size}"
        )

    leaf_edges = y_min + np.arange(machine.mlc_num_leaf_pairs + 1) * machine.mlc_leaf_thickness_mm
    for leaf_idx in range(machine.mlc_num_leaf_pairs):
        if left[leaf_idx] <= 0 and right[leaf_idx] <= 0:
            continue
        si_open = (si >= leaf_edges[leaf_idx]) & (si < leaf_edges[leaf_idx + 1])
        lo = -max(left[leaf_idx], right[leaf_idx])
        hi = -min(left[leaf_idx], right[leaf_idx])
        lateral_open = (lateral >= lo) & (lateral <= hi)
        mask[np.ix_(si_open, lateral_open)] = 1.0

    return mask, si, lateral


def project_mlc_to_bev(
    cp: dict,
    machine: PhotonMachineGeometry,
    shape_dsl: Tuple[int, int, int],
    spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    aperture_2d, _, _ = build_mlc_aperture_isoplane(cp, machine)

    depth_n, si_n, lateral_n = shape_dsl
    depth = (np.arange(depth_n, dtype=np.float32) - (depth_n - 1) / 2.0) * spacing_mm
    si = (np.arange(si_n, dtype=np.float32) - (si_n - 1) / 2.0) * spacing_mm
    lateral = (np.arange(lateral_n, dtype=np.float32) - (lateral_n - 1) / 2.0) * spacing_mm

    d_grid, si_grid, lat_grid = np.meshgrid(depth, si, lateral, indexing="ij")
    denom = machine.sad_mm + d_grid
    scale_to_isoplane = np.divide(
        machine.sad_mm,
        denom,
        out=np.zeros_like(d_grid, dtype=np.float32),
        where=np.abs(denom) > 1e-6,
    )
    si_iso = si_grid * scale_to_isoplane
    lat_iso = lat_grid * scale_to_isoplane

    x_min, x_max = machine.jaw_x_mm
    y_min, y_max = machine.jaw_y_mm
    leaf_idx = np.floor((si_iso - y_min) / machine.mlc_leaf_thickness_mm).astype(np.int32)
    valid = (
        (lat_iso >= x_min)
        & (lat_iso <= x_max)
        & (si_iso >= y_min)
        & (si_iso <= y_max)
        & (leaf_idx >= 0)
        & (leaf_idx < machine.mlc_num_leaf_pairs)
    )

    left = np.asarray(cp["mlc_left_int_mm"], dtype=np.float32)
    right = np.asarray(cp["mlc_right_int_mm"], dtype=np.float32)
    open_left = np.zeros_like(lat_iso, dtype=np.float32)
    open_right = np.zeros_like(lat_iso, dtype=np.float32)
    open_left[valid] = left[leaf_idx[valid]]
    open_right[valid] = right[leaf_idx[valid]]
    leaf_lo = -np.maximum(open_left, open_right)
    leaf_hi = -np.minimum(open_left, open_right)

    projection = (
        valid
        & (leaf_hi > leaf_lo)
        & (lat_iso >= leaf_lo)
        & (lat_iso <= leaf_hi)
    ).astype(np.float32)
    return projection, aperture_2d


def make_bev_sitk(
    array_dsl: np.ndarray,
    geometry: BeamGeometry,
    spacing_mm: float,
) -> sitk.Image:
    depth_n, si_n, lateral_n = array_dsl.shape

    depth_axis = np.asarray(geometry.beam_axis_xyz, dtype=np.float64)
    si_axis = np.asarray(geometry.si_axis_xyz, dtype=np.float64)
    lateral_axis = np.asarray(geometry.lateral_axis_xyz, dtype=np.float64)
    iso = np.asarray(geometry.iso_center_xyz, dtype=np.float64)

    origin = (
        iso
        - ((depth_n - 1) / 2.0) * spacing_mm * depth_axis
        - ((si_n - 1) / 2.0) * spacing_mm * si_axis
        - ((lateral_n - 1) / 2.0) * spacing_mm * lateral_axis
    )

    # NIfTI/SimpleITK array input is (z, y, x). We expose image axes as:
    # x=lateral, y=superior/inferior, z=depth.
    direction = np.column_stack((lateral_axis, si_axis, depth_axis))
    image = sitk.GetImageFromArray(array_dsl.astype(np.float32))
    image.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    image.SetOrigin(tuple(float(v) for v in origin))
    image.SetDirection(tuple(float(v) for v in direction.reshape(-1)))
    return image


def sample_output_dir(output_dir: Path, task: SampleTask) -> Path:
    return output_dir / task.group_name / task.key


def sample_is_complete(sample_dir: Path) -> bool:
    return all((sample_dir / name).exists() for name in ("ct.h5", "proj.h5", "dose.h5", "metadata.json"))


def write_array_hdf5(path: Path, array: np.ndarray, compression: str | None) -> None:
    kwargs = {"compression": compression} if compression else {}
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=array.astype(np.float32), dtype=np.float32, **kwargs)


def write_sample_folder(
    output_dir: Path,
    task: SampleTask,
    geometry: BeamGeometry,
    machine: PhotonMachineGeometry,
    shape_dsl: Tuple[int, int, int],
    spacing_mm: float,
    compression: str | None,
    ct_bev: np.ndarray,
    mlc_projection: np.ndarray,
    dose_bev: np.ndarray,
    aperture_2d: np.ndarray,
    write_nifti: bool,
) -> Path:
    sample_dir = sample_output_dir(output_dir, task)
    sample_dir.mkdir(parents=True, exist_ok=True)

    write_array_hdf5(sample_dir / "ct.h5", ct_bev, compression)
    write_array_hdf5(sample_dir / "proj.h5", mlc_projection, compression)
    write_array_hdf5(sample_dir / "dose.h5", dose_bev, compression)

    if write_nifti:
        for name, array in (("ct", ct_bev), ("dose", dose_bev), ("proj", mlc_projection)):
            sitk.WriteImage(
                make_bev_sitk(array, geometry, spacing_mm),
                str(sample_dir / f"{name}.nii.gz"),
            )
        aperture_img = sitk.GetImageFromArray(aperture_2d.astype(np.float32))
        aperture_img.SetSpacing((1.0, 1.0))
        sitk.WriteImage(aperture_img, str(sample_dir / "mlc_aperture_isoplane.nii.gz"))

    metadata = {
        "key": task.key,
        "group": task.group_name,
        "patient_id": task.patient_id,
        "beam_index": task.beam_index,
        "cp_index": task.cp_index,
        "ct_path": str(task.ct_path),
        "dose_path": str(task.dose_path),
        "shape_depth_si_lateral": list(shape_dsl),
        "spacing_mm": spacing_mm,
        "geometry": {
            "iso_center_xyz": list(geometry.iso_center_xyz),
            "gantry_angle_deg": geometry.gantry_angle_deg,
            "beam_axis_xyz": list(geometry.beam_axis_xyz),
            "lateral_axis_xyz": list(geometry.lateral_axis_xyz),
            "si_axis_xyz": list(geometry.si_axis_xyz),
        },
        "machine": {
            "sad_mm": machine.sad_mm,
            "jaw_x_mm": list(machine.jaw_x_mm),
            "jaw_y_mm": list(machine.jaw_y_mm),
            "mlc_num_leaf_pairs": machine.mlc_num_leaf_pairs,
            "mlc_leaf_thickness_mm": machine.mlc_leaf_thickness_mm,
        },
        "files": {
            "ct": "ct.h5",
            "proj": "proj.h5",
            "dose": "dose.h5",
        },
        "hdf5_dataset_name": "data",
        "dose_min": float(dose_bev.min()),
        "dose_max": float(dose_bev.max()),
        "projection_open_fraction": float(mlc_projection.mean()),
        "nifti_written": write_nifti,
    }
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return sample_dir


def write_dataset_manifest(
    output_dir: Path,
    training_root: Path,
    machine: PhotonMachineGeometry,
    shape_dsl: Tuple[int, int, int],
    spacing_mm: float,
    interpolation: str,
    compression: str | None,
    write_nifti: bool,
    tasks: list[SampleTask],
    processed: list[dict],
    skipped: list[dict],
    patient_summary: dict[str, dict[str, int | str]],
    discovered_count_before_resume: int,
    resume_from_index: int | None,
    resume_from_key: str | None,
    resume_message: str | None,
    overwrite_existing: bool,
) -> None:
    group_counts = {
        "train": sum(1 for task in tasks if task.group_name == "train"),
        "validation": sum(1 for task in tasks if task.group_name == "validation"),
    }
    manifest = {
        "description": "BEV preprocessing output for DoseRAD2026 photon segment dose calculation.",
        "training_root": str(training_root),
        "output_dir": str(output_dir),
        "shape_depth_si_lateral": list(shape_dsl),
        "spacing_mm": spacing_mm,
        "interpolation": interpolation,
        "compression": compression,
        "hdf5_layout": "One folder per sample under train/ or validation/. Each sample folder contains ct.h5, proj.h5, dose.h5, and metadata.json.",
        "hdf5_dataset_name": "data",
        "sample_key_format": "{patient_id}_B{beam_index}_CP{cp_index:03d}",
        "proj_dataset_note": "Divergent MLC aperture projection. Aperture is assumed to lie on the isocentre plane.",
        "nifti_preview_written": write_nifti,
        "machine": {
            "sad_mm": machine.sad_mm,
            "jaw_x_mm": list(machine.jaw_x_mm),
            "jaw_y_mm": list(machine.jaw_y_mm),
            "mlc_num_leaf_pairs": machine.mlc_num_leaf_pairs,
            "mlc_leaf_thickness_mm": machine.mlc_leaf_thickness_mm,
        },
        "sample_counts": {
            "selected_after_resume": len(tasks),
            "discovered_before_resume": discovered_count_before_resume,
            **group_counts,
            "patients": len({task.patient_id for task in tasks}),
            "written_this_run": len(processed),
            "skipped_existing_this_run": len(skipped),
        },
        "resume": {
            "resume_from_index": resume_from_index,
            "resume_from_key": resume_from_key,
            "message": resume_message,
            "overwrite_existing": overwrite_existing,
        },
        "patient_summary": patient_summary,
        "written_samples": processed,
        "skipped_samples": skipped,
    }
    with open(output_dir / "bev_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def parse_shape(value: str) -> Tuple[int, int, int]:
    parts = tuple(int(v) for v in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be depth,si,lateral, for example 256,200,200")
    return parts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build BEV CT/projection/dose HDF5 files for DoseRAD2026 photon segments."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient-id", action="append", default=None,
                        help="Patient id to process. Can be repeated. Default: all patients.")
    parser.add_argument("--beam-index", type=int, default=None,
                        help="Optional beam filter. Default: all beams.")
    parser.add_argument("--cp-index", type=int, default=None,
                        help="Optional control-point filter. Default: all CPs.")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Optional smoke-test limit after sample discovery.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional smoke-test limit after sample discovery.")
    parser.add_argument("--resume-from-index", type=int, default=None,
                        help="Resume from this 1-based sample index in the discovered task order, inclusive.")
    parser.add_argument("--resume-from-key", type=str, default=None,
                        help="Resume from this sample key, for example 1THB076_B1_CP064, inclusive.")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="Patient-level validation split from the sorted patient list. Default: 0.2")
    parser.add_argument("--validation-patient-id", action="append", default=None,
                        help="Explicit validation patient id. Can be repeated; overrides --validation-fraction.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print discovered patients/sample counts; do not write HDF5 files.")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="Rewrite sample folders that already contain ct.h5, proj.h5, dose.h5, and metadata.json.")
    parser.add_argument("--shape", type=parse_shape, default=DEFAULT_SHAPE_DSL,
                        help="Output shape as depth,superior-inferior,lateral. Default: 256,200,200")
    parser.add_argument("--spacing-mm", type=float, default=DEFAULT_SPACING_MM)
    parser.add_argument("--interp", choices=["linear", "nearest", "cubic"], default="linear")
    parser.add_argument("--write-nifti", action="store_true",
                        help="Also write BEV CT/dose/proj .nii.gz previews for review.")
    parser.add_argument("--nifti-max", type=int, default=20,
                        help="Maximum number of samples to write as NIfTI when --write-nifti is set. Default: 20")
    parser.add_argument("--no-write-nifti", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    training_root = photon_training_root(args.data_root)
    machine = load_photon_machine_geometry(training_root)
    selected_patient_ids = set(args.patient_id) if args.patient_id else None
    validation_patient_ids = set(args.validation_patient_id) if args.validation_patient_id else None
    tasks = discover_samples(
        training_root,
        selected_patient_ids=selected_patient_ids,
        selected_beam_index=args.beam_index,
        selected_cp_index=args.cp_index,
        validation_fraction=args.validation_fraction,
        validation_patient_ids=validation_patient_ids,
    )
    tasks = limit_samples(tasks, args.max_patients, args.max_samples)
    discovered_count_before_resume = len(tasks)
    tasks, resume_start_offset, resume_message = resume_tasks(tasks, args.resume_from_index, args.resume_from_key)
    if not tasks:
        raise RuntimeError("No samples matched the requested filters.")

    interp_order = {"nearest": 0, "linear": 1, "cubic": 3}[args.interp]
    compression = None if args.compression == "none" else args.compression
    write_nifti = args.write_nifti and not args.no_write_nifti

    active_patient_ids = {task.patient_id for task in tasks}
    patient_summary = {
        patient_id: {
            "samples": sum(1 for task in tasks if task.patient_id == patient_id),
            "beams": len({task.beam_index for task in tasks if task.patient_id == patient_id}),
            "group": next(task.group_name for task in tasks if task.patient_id == patient_id),
        }
        for patient_id in sorted(active_patient_ids)
    }
    group_counts = {
        "train": sum(1 for task in tasks if task.group_name == "train"),
        "validation": sum(1 for task in tasks if task.group_name == "validation"),
    }

    print(f"Photon training root: {training_root}")
    print(f"Output dir:           {args.output_dir}")
    print(f"Shape DSL:            {args.shape}")
    print(f"Spacing:              {args.spacing_mm} mm")
    print(f"SAD / jaws:           {machine.sad_mm} mm / X{machine.jaw_x_mm} Y{machine.jaw_y_mm}")
    print(
        f"Samples:              {len(tasks)} selected after resume / "
        f"{discovered_count_before_resume} discovered before resume / "
        f"{group_counts['train']} train / {group_counts['validation']} validation"
    )
    if resume_message is not None:
        print(f"Resume:               {resume_message}")
    print("Patients to process:")
    for patient_id, summary in patient_summary.items():
        print(
            f"  {patient_id}: {summary['samples']} samples, "
            f"{summary['beams']} beams, group={summary['group']}"
        )
    if args.dry_run:
        print("Dry run only; no HDF5 files were written.")
        return
    print("Output layout:        <output-dir>/<train|validation>/<sample_key>/{ct.h5,proj.h5,dose.h5,metadata.json}")
    overwrite_existing = args.overwrite_existing or resume_message is not None
    legacy_files = [
        args.output_dir / "ct_dataset.h5",
        args.output_dir / "proj_dataset.h5",
        args.output_dir / "dose_dataset.h5",
    ]
    existing_legacy = [path.name for path in legacy_files if path.exists()]
    if existing_legacy:
        print(f"Legacy aggregate HDF5 files found and ignored: {', '.join(existing_legacy)}")
    if overwrite_existing:
        print("Existing complete sample folders will be overwritten.")
        if resume_message is not None and not args.overwrite_existing:
            print("Resume mode forces overwrite from the resume point for data safety.")
    else:
        print("Existing complete sample folders will be skipped.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed: list[dict] = []
    skipped: list[dict] = []
    current_patient_id: str | None = None
    ct_img: sitk.Image | None = None
    ct_array: np.ndarray | None = None
    nifti_written = 0

    for index, task in enumerate(tasks, start=resume_start_offset + 1):
        sample_dir = sample_output_dir(args.output_dir, task)
        if sample_is_complete(sample_dir) and not overwrite_existing:
            print(f"[{index}/{discovered_count_before_resume}] {task.key} -> skip existing", flush=True)
            skipped.append({"key": task.key, "group": task.group_name, "sample_dir": str(sample_dir)})
            continue

        if task.patient_id != current_patient_id:
            print(f"\nLoading patient {task.patient_id}: {task.ct_path}")
            ct_img = read_float_image(task.ct_path)
            ct_array = sitk.GetArrayFromImage(ct_img).astype(np.float32)
            current_patient_id = task.patient_id

        assert ct_img is not None
        assert ct_array is not None
        geometry = geometry_from_beam_cp(task.beam, task.cp)

        print(f"[{index}/{discovered_count_before_resume}] {task.key} -> {sample_dir}", flush=True)
        dose_img = resample_dose_to_ct_if_needed(read_float_image(task.dose_path), ct_img)
        dose_array = sitk.GetArrayFromImage(dose_img).astype(np.float32)

        ct_bev = resample_array_to_bev(
            ct_array,
            ct_img,
            geometry,
            shape_dsl=args.shape,
            spacing_mm=args.spacing_mm,
            order=interp_order,
            cval=-1024.0,
        )
        ct_bev = np.clip(ct_bev, -1024.0, 3071.0)

        dose_bev = resample_array_to_bev(
            dose_array,
            ct_img,
            geometry,
            shape_dsl=args.shape,
            spacing_mm=args.spacing_mm,
            order=interp_order,
            cval=0.0,
        )
        dose_bev = np.clip(dose_bev, 0.0, None)

        mlc_projection, aperture_2d = project_mlc_to_bev(
            task.cp,
            machine,
            shape_dsl=args.shape,
            spacing_mm=args.spacing_mm,
        )

        sample_write_nifti = write_nifti and nifti_written < args.nifti_max
        written_dir = write_sample_folder(
            args.output_dir,
            task,
            geometry,
            machine,
            args.shape,
            args.spacing_mm,
            compression,
            ct_bev,
            mlc_projection,
            dose_bev,
            aperture_2d,
            sample_write_nifti,
        )
        if sample_write_nifti:
            nifti_written += 1

        processed.append(
            {
                "key": task.key,
                "group": task.group_name,
                "sample_dir": str(written_dir),
                "patient_id": task.patient_id,
                "beam_index": task.beam_index,
                "cp_index": task.cp_index,
                "gantry_angle_deg": geometry.gantry_angle_deg,
                "ct_path": str(task.ct_path),
                "dose_path": str(task.dose_path),
                "dose_min": float(dose_bev.min()),
                "dose_max": float(dose_bev.max()),
                "projection_open_fraction": float(mlc_projection.mean()),
            }
        )

    write_dataset_manifest(
        args.output_dir,
        training_root=training_root,
        machine=machine,
        shape_dsl=args.shape,
        spacing_mm=args.spacing_mm,
        interpolation=args.interp,
        compression=compression,
        write_nifti=write_nifti,
        tasks=tasks,
        processed=processed,
        skipped=skipped,
        patient_summary=patient_summary,
        discovered_count_before_resume=discovered_count_before_resume,
        resume_from_index=args.resume_from_index,
        resume_from_key=args.resume_from_key,
        resume_message=resume_message,
        overwrite_existing=overwrite_existing,
    )

    print("\nDone.")
    print(f"Wrote {len(processed)} samples to {args.output_dir}")
    print(f"Skipped existing complete samples: {len(skipped)}")
    print(f"NIfTI previews written: {nifti_written}")


if __name__ == "__main__":
    main()
