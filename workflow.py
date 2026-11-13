#!/usr/bin/env python3
"""Portable entry points for the submitted CT-to-dose implementation."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("geometry", "preprocess", "train"))
    parser.add_argument("--data-root", type=Path, required=True, help="DoseRAD2026 directory containing photon/training")
    parser.add_argument("--preprocessed-root", type=Path, default=ROOT / "data/bev")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.25e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-patients", default="1ABB006,1THB120")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not (data_root / "photon/training/beam_parameters.json").is_file():
        parser.error("--data-root must contain photon/training/beam_parameters.json")
    if args.stage == "geometry":
        module = load("1.2_Precompute_Geometry.py", "geometry")
        module.DEFAULT_DATA_ROOT = data_root
    elif args.stage == "preprocess":
        module = load("1.3_New_Preprocessing_bspline.py", "preprocess")
        module.DATA_ROOT = data_root
        module.OUTPUT_ROOT = args.preprocessed_root.resolve()
        module.DEVICE = args.device
        module.MAX_SAMPLES = args.max_samples
        module.VALIDATION_PATIENT_IDS = set(args.validation_patients.split(","))
    else:
        module = load("5_train_LSTM_Res_Codex.py", "training")
        module.PREPROCESSED_ROOT = args.preprocessed_root.resolve()
        module.RAW_DOSE_ROOT = data_root / "photon/training"
        module.MODEL_CODEX_ROOT = args.output_root.resolve()
        module.DEVICE = args.device
        module.EPOCHS = args.epochs
        module.LEARNING_RATE = args.lr
        module.NUM_WORKERS = args.workers
        module.PERSISTENT_WORKERS = args.workers > 0
        module.MAX_SAMPLES = args.max_samples
        module.RESUME_FROM_CHECKPOINT = args.resume is not None
        module.RESUME_CHECKPOINT_PATH = args.resume.resolve() if args.resume else None
        module.RESUME_RUN_ID = None
    module.main()


if __name__ == "__main__":
    main()
