#!/usr/bin/env python3
"""Evaluate CT-to-dose checkpoints against raw held-out CP dose labels."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
import SimpleITK as sitk

from workflow import ROOT, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--patients", default="1ABB006,1THB120")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 <= args.threshold < 1:
        parser.error("--threshold must be in [0,1)")
    args.output.mkdir(parents=True, exist_ok=False)
    # The submitted loader expects this conventional filename.
    bundle = args.output / "bundle"
    bundle.mkdir()
    (bundle / "best_model.pt").symlink_to(args.checkpoint.resolve())
    for name in ("5_train_LSTM_Res_Codex.py", "1_Preprocessing.py", "1.3_New_Preprocessing.py", "model", "precomputed_geometry_v1_2"):
        (bundle / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    resources = bundle / "resources"
    resources.mkdir()
    raw_root = args.data_root.resolve() / "photon/training"
    shutil.copy2(raw_root / "beam_parameters.json", resources / "beam_parameters.json")
    os.environ.update(MODEL_PATH=str(bundle.resolve()), DEVICE=args.device,
                     ENABLE_INIT_WARMUP="0", ENABLE_TORCH_COMPILE="0",
                     ENABLE_CUDA_GRAPH="0", ENABLE_TORCHSCRIPT_TRACE="0")
    inference = load("inference.py", "submission_inference")
    algorithm = inference.DoseRadResAlgorithm()
    rows = []

    def record(level, name, reference, prediction):
        mask = reference > args.threshold * float(reference.max())
        error = np.abs(prediction - reference)
        rows.append(dict(level=level, name=name, voxels=int(mask.sum()),
                         mae=float(error[mask].mean()) if mask.any() else None,
                         reference_peak=float(reference.max())))

    for patient in args.patients.split(","):
        case = raw_root / patient.strip()
        metadata = json.loads((case / f"{case.name}.json").read_text())
        ct = sitk.ReadImage(str(case / "image/ct.mha"), sitk.sitkFloat32)
        cache = inference.build_patient_cache(0, ct, algorithm.hu, algorithm.density_curve, algorithm.device)
        for beam in metadata["beams"]:
            ref_sum = pred_sum = None
            for cp in beam["control_points"]:
                name = f"{case.name}_B{beam['beam_idx']}_CP{int(cp['cp_idx']):03d}"
                ref_img = sitk.ReadImage(str(case / "dose" / f"Dose_B{beam['beam_idx']}_CP{int(cp['cp_idx']):03d}.mha"))
                if ref_img.GetSize() != ct.GetSize() or not np.allclose(ref_img.GetSpacing(), ct.GetSpacing()) or not np.allclose(ref_img.GetOrigin(), ct.GetOrigin()) or not np.allclose(ref_img.GetDirection(), ct.GetDirection()):
                    raise ValueError(f"CT/dose geometry mismatch: {name}")
                ref = sitk.GetArrayFromImage(ref_img).astype(np.float32)
                pred, _ = algorithm.predict_control_point(cache, ct, beam, cp, minimum_cutoff=0.0)
                record("cp", name, ref, pred)
                ref_sum = ref.copy() if ref_sum is None else ref_sum + ref
                pred_sum = pred.copy() if pred_sum is None else pred_sum + pred
                print(name, rows[-1]["mae"], flush=True)
            if ref_sum is not None:
                record("plan", f"{case.name}_B{beam['beam_idx']}", ref_sum, pred_sum)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["level", "name", "voxels", "mae", "reference_peak"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "threshold": args.threshold}
    for level in ("cp", "plan"):
        values = [row["mae"] for row in rows if row["level"] == level and row["mae"] is not None]
        summary[level] = {"count": len(values), "mean_mae": float(np.mean(values)) if values else None}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
