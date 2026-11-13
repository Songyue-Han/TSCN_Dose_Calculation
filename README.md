# TSCN_Dose_Calculation

Research code for the photon CT submission to DoseRAD2026. The model combines
2D CNN feature extraction, a depth-wise temporal convolutional network (TCN),
and separate BEV dose and residual heads. Both outputs are resampled to the
patient grid and added for the final dose prediction.

This release contains source code only. Obtain DoseRAD data separately;
checkpoints, patient images, generated BEV data and submission container images
are not included. This V1 covers the CT-to-dose workflow.

## Environment

Use Python 3.10 or later. The submission used PyTorch 2.8.0 with CUDA 12.8.
Install the appropriate PyTorch build for your GPU, then:

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended. Full patient-grid training and BEV preprocessing
are memory and disk intensive; start with a small data subset when setting up.

## Data

Pass the directory containing `photon/training` as `--data-root`:

```text
DoseRAD2026/
  photon/training/
    beam_parameters.json
    1ABB006/
      1ABB006.json
      image/ct.mha
      dose/Dose_B1_CP000.mha
      ...
    ...
```

Use the dataset's original patient metadata, machine calibration and per-CP dose
labels. Default held-out patients are `1ABB006` and `1THB120`. Preprocessing
assigns all CPs of each patient to one split; validation patients must be present.

## Preprocess

Run commands from this folder. Replace `/path/to/DoseRAD2026` with your data path.

```bash
python workflow.py geometry --data-root /path/to/DoseRAD2026
python workflow.py preprocess --data-root /path/to/DoseRAD2026 --preprocessed-root data/bev --device cuda:0
```

Geometry is generated locally in `precomputed_geometry_v1_2`. Preprocessing
writes `train/` and `validation/` CP directories with `maps.h5` and metadata.
Channels include CT, aperture projection, inverse-square and water-equivalent
depth (WED). Offline preprocessing uses the B-spline CT sampling implementation
used by the later training workflow. The submitted online predictor retains
its own sampling implementation in `1.3_New_Preprocessing.py`.

Use `--validation-patients ID1,ID2` to change the held-out split. Use a new
preprocessing output directory when changing splits; do not mix old split files.
`--max-samples N` limits preprocessing for setup checks; choose enough samples
from both splits before attempting training.

## Train And Validate

```bash
python workflow.py train --data-root /path/to/DoseRAD2026 --preprocessed-root data/bev --output-root runs --epochs 1000 --lr 0.000125 --workers 4 --device cuda:0
```

This starts from random initialization using the final submission architecture
and final loss configuration. Validation runs every epoch on the preprocessed
held-out split. Training also reads original patient-grid dose labels from the
raw data directory. Outputs include timestamped runs, checkpoints, split
manifests, logs and TensorBoard events. Inspect them with:

```bash
tensorboard --logdir runs
```

The main objective is patient-grid regional MAE (global, MLC interior,
penumbra and reference-dose bins), with a small final-dose IDD auxiliary term.
The final configuration disables direct BEV branch supervision; gradients from
patient-grid supervision still reach the BEV branch. CT is scaled by 0.001,
WED by 1/500, and dose targets by 10000; inference divides by the saved dose scale.
See the constants in `5_train_LSTM_Res_Codex.py` for the detailed weights.

Resume training, restoring the saved optimizer and scheduler:

```bash
python workflow.py train --data-root /path/to/DoseRAD2026 --preprocessed-root data/bev --output-root runs --resume runs/RUN/checkpoints/epoch_0050.pt --epochs 1000
```

`--epochs` is the final epoch index, including previously completed epochs.
The saved optimizer learning rate is restored when resuming.

## Evaluate A Trained Checkpoint

```bash
python evaluate.py --data-root /path/to/DoseRAD2026 --checkpoint runs/RUN/checkpoints/best_model.pt --patients 1ABB006,1THB120 --output outputs/validation_run --device cuda:0
```

Use a new output directory per evaluation. The script runs the submitted
predictor on every selected CP, then sums CP predictions and references per
beam (called a plan here). It writes `metrics.csv` and `summary.json` with CP
and plan MAE over reference dose >10% of that CP/plan's peak. Set `--threshold`
to change this fraction. Dose metrics retain the original dataset's dose units.
This lightweight assessment does not reproduce the challenge's full scoring.

## Code Map

| File | Purpose |
| --- | --- |
| `workflow.py` | Portable paths and geometry/preprocessing/training entry points |
| `1_Preprocessing.py` | Raw dataset discovery, geometry and sampling helpers |
| `1.2_Precompute_Geometry.py` | Shared BEV master-grid geometry |
| `1.3_New_Preprocessing_bspline.py` | Offline training sample generation |
| `5_train_LSTM_Res_Codex.py` | Dataset, losses, training, validation and checkpoints |
| `model/CNN_TCN_Res_Codex/model.py` | CNN/TCN, conditioning, decoder and heads |
| `model/CNN_TCN_Res_Codex/cubic_bspline.py` | Output interpolation helpers |
| `0_gamma_gpu.py` | Gamma calculation for periodic raw-grid validation |
| `inference.py` | Submitted CT predictor and challenge-format output writer |
| `1.3_New_Preprocessing.py` | Online submission preprocessing helpers |
| `evaluate.py` | Held-out CP/beam evaluation on raw data |

Use `workflow.py` as the entry point to configure paths and training options.
No code or model from the parent project directory is required.
