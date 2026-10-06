# Laser-Speckle-ML-Controller-with-DMD
Undergraduate dissertation project - Data-driven control of a DMD laser projection system through a diffuser, using Machine Learning through MATLAB.

Recovering binary DMD patterns from laser speckle using MLPs and PPO in MATLAB, CPU-only.

A laser is modulated by a Digital Micromirror Device (DMD), scattered by a diffuser and captured by a camera. A model learns the inverse mapping, predicting the 32×32 binary DMD pattern from the speckle image alone. A shallow MLP and a PPO agent are compared.

## Pipeline

| Folder | Scripts |
|---|---|
| `01_calibration/` | `exposure_calibration_test.m`, `speckle_contrast.m` |
| `02_data_acquisition/` | `rpatt.m` (needs the DMD and camera) |
| `03_preprocessing/` | `crop.m`, `flip.m`, `shift1.m` (run in this order, once each) |
| `04_models/` | `slnn.m` (MLPs), `ppo.m` |
| `05_evaluation/` | `model_eval3.m`, `subsystem_test.m` |

## Results

Evaluated on 1,000 unseen test samples (an all-off baseline scores 57.75% accuracy).

![Side-by-side comparison](results/figures/compare_side_by_side.png)

## Requirements

MATLAB R2025b with the Deep Learning, Reinforcement Learning, Image Processing and Statistics Toolboxes (plus Image Acquisition for stages 01 and 02). No GPU needed.

Optical bench originally built by Gabriel Pacheco.
