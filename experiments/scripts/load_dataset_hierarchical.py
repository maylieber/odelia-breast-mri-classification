"""
Parallel dataloader builder for the two-stage (hierarchical) cascade
classification experiment. Mirrors load_dataset.py's create_dataloaders()
exactly (same paths, batch size, augmentation, class-weight mechanism via
sklearn's "balanced" compute_class_weight) but builds two independent
sets of loaders:

    Stage 1 (all patients, same train/val/test split as the reference configuration):
        No lesion (0) vs. Lesion (1) -- Benign+Malignant collapsed.

    Stage 2 (train/val/test split restricted to lesion-positive patients
    only, i.e. the No-lesion rows are dropped before splitting):
        Benign (0) vs. Malignant (1).

load_dataset.py / dataset.py are untouched.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

# so root-level modules (augmentations.py, etc.) and data files resolve
# regardless of the working directory this script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_hierarchical import ODELIADatasetRemapped
from augmentations import train_augmentations

# ==========================================================
# Same settings as load_dataset.py
# ==========================================================

PROCESSED_ROOT = Path(r"D:\ODELIA_processed")
PROCESSED_MASTER_TABLE = PROJECT_ROOT / "processed_master_dataset.csv"

BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = False

STAGE1_LABEL_MAP = {0: 0, 1: 1, 2: 1}  # No lesion=0, Lesion=1
STAGE2_LABEL_MAP = {1: 0, 2: 1}        # Benign=0, Malignant=1


def _class_weights(labels, num_classes):

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels
    )

    return torch.tensor(weights, dtype=torch.float32)


def _make_loaders(train_dataset, val_dataset, test_dataset):

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader, test_loader


def create_stage1_dataloaders():
    """
    No lesion vs. Lesion, on the full dataset (same train/val/test split
    as the reference configuration).
    """

    df = pd.read_csv(PROCESSED_MASTER_TABLE)

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "val"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    train_dataset = ODELIADatasetRemapped(
        train_df, PROCESSED_ROOT, STAGE1_LABEL_MAP,
        transform=train_augmentations()
    )
    val_dataset = ODELIADatasetRemapped(
        val_df, PROCESSED_ROOT, STAGE1_LABEL_MAP
    )
    test_dataset = ODELIADatasetRemapped(
        test_df, PROCESSED_ROOT, STAGE1_LABEL_MAP
    )

    train_loader, val_loader, test_loader = _make_loaders(train_dataset, val_dataset, test_dataset)

    train_labels = train_df["Lesion"].map(STAGE1_LABEL_MAP).values
    class_weights = _class_weights(train_labels, num_classes=2)

    return train_loader, val_loader, test_loader, class_weights


def create_stage2_dataloaders():
    """
    Benign vs. Malignant, restricted to lesion-positive patients only.
    No-lesion rows are dropped BEFORE splitting, so this stage's
    train/val/test sets are subsets of the reference configuration's
    original split (same patients, just missing the Lesion=0 rows), not a
    fresh split.
    """

    df = pd.read_csv(PROCESSED_MASTER_TABLE)
    df = df[df["Lesion"].isin([1, 2])].reset_index(drop=True)

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "val"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    train_dataset = ODELIADatasetRemapped(
        train_df, PROCESSED_ROOT, STAGE2_LABEL_MAP,
        transform=train_augmentations()
    )
    val_dataset = ODELIADatasetRemapped(
        val_df, PROCESSED_ROOT, STAGE2_LABEL_MAP
    )
    test_dataset = ODELIADatasetRemapped(
        test_df, PROCESSED_ROOT, STAGE2_LABEL_MAP
    )

    train_loader, val_loader, test_loader = _make_loaders(train_dataset, val_dataset, test_dataset)

    train_labels = train_df["Lesion"].map(STAGE2_LABEL_MAP).values
    class_weights = _class_weights(train_labels, num_classes=2)

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":

    s1_train, s1_val, s1_test, s1_weights = create_stage1_dataloaders()
    s2_train, s2_val, s2_test, s2_weights = create_stage2_dataloaders()

    print("Stage 1 (No lesion vs. Lesion)")
    print(f"  train/val/test patients: {len(s1_train.dataset)}/{len(s1_val.dataset)}/{len(s1_test.dataset)}")
    print(f"  class weights: {s1_weights}")

    print("Stage 2 (Benign vs. Malignant)")
    print(f"  train/val/test patients: {len(s2_train.dataset)}/{len(s2_val.dataset)}/{len(s2_test.dataset)}")
    print(f"  class weights: {s2_weights}")