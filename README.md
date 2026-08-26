# ODELIA Breast MRI Lesion Classification

3-class patient-level classification from breast MRI: **No lesion** / **Benign** / **Malignant**,
using the ODELIA Challenge 2025 dataset. Per-slice ResNet18 features (ImageNet-pretrained,
only `layer4` fine-tuned), attention-pooled across 32 slices per patient, classified by a
small MLP head.

This submission's reference configuration: augmentation on, batch size 16, `layer4`
unfrozen, `lr=1e-5`, `weight_decay=1e-4`, dropout 0.5, 15 epochs, T2 + Pre + Post-contrast-1
modalities. Reported test performance: **55.3% accuracy, macro AUC 0.689**.

Every module — in both the main pipeline and `experiments/` — is documented with clear,
purpose-focused docstrings and comments: what each function does, and why non-obvious
design decisions (e.g. checkpoint-compatible naming, evaluation logic) were made.

## Layout

```
augmentations.py, compute_class_weights.py, dataset.py, load_dataset.py,
master_table.py, model.py, preprocessing.py, roi.py, visualization.py   core pipeline modules
download_dataset.py, setup_dataset.py                                    one-time dataset setup
train.py                                                                  trains the reference configuration
inference.py                                                              runs the trained model, no retraining needed
checkpoints/best_model_run.pt                                            trained weights (the deliverable)
figures/                                                                  loss curve, confusion matrix, ROC curve
master_dataset.csv, processed_master_dataset.csv                         dataset tables
experiments/                                                              archived alternative runs (see below)
```

`experiments/` holds every other configuration that was tried (other backbones, other
hyperparameters, the leave-one-institution-out cross-validation, the two-stage cascade,
etc.) — their scripts (`experiments/scripts/`) and trained checkpoints
(`experiments/checkpoints/`), kept for provenance. Each script resolves the shared root
modules (`dataset.py`, `model.py`, etc.) and its data/output paths relative to its own
location, so it can be run directly (`python experiments/scripts/<file>.py`) from any
working directory — outputs land in `experiments/checkpoints/` and `experiments/figures/`.

## Setup

```
pip install -r requirements.txt
```

A few scripts have machine-specific absolute paths that need updating before running on a
new machine:
- `download_dataset.py`: `output_root` (where raw downloads land) — also needs an `HF_TOKEN`
  environment variable set to a Hugging Face access token with access to the
  `ODELIA-AI/ODELIA-Challenge-2025` dataset (**do not hardcode the token in the file**).
- `setup_dataset.py`: `ORIGINAL_DATASET`, `PROCESSED_DATASET`
- `load_dataset.py`: `PROCESSED_ROOT`

## Pipeline

```
download_dataset.py → master_table.py / setup_dataset.py → preprocessing.py / roi.py
    → dataset.py → load_dataset.py → model.py → train.py → inference.py
```

## Training

```
python train.py
```

Trains the reference configuration from scratch and writes `checkpoints/best_model_run.pt`
plus `figures/loss_curve_run.png`, `figures/confusion_matrix_run.png`,
`figures/roc_curve_run.png`. Test-set evaluation reports per-class sensitivity/specificity/
AUC plus macro-, weighted-, and micro-average AUC.

## Inference (no retraining needed)

Uses the included `checkpoints/best_model_run.pt`.

Single preprocessed patient (a folder containing `T2.nii.gz`, `Pre.nii.gz`, `Post1.nii.gz`,
already ROI-cropped and resized to 128x128x32 by `preprocessing.py`):

```
python inference.py --patient_dir path/to/patient_folder
```

Reproduce the reported test-set metrics on the full processed test split — per-class
sensitivity/specificity/AUC plus macro-, weighted-, and micro-average AUC:

```
python inference.py --test_set
```
