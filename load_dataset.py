"""
load_dataset.py

Utility for loading the processed ODELIA dataset.

Student B only needs to modify the two paths below.

Returns:
    train_loader
    validation_loader
    test_loader
"""

from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ODELIADataset
from compute_class_weights import compute_weights
from augmentations import train_augmentations

# ==========================================================
# USER SETTINGS
# ==========================================================

# Root directory of the processed images
PROCESSED_ROOT = Path(r"D:\ODELIA_processed")

# Processed master table (anchored to this file's location, not the
# working directory, so callers importing this module or its
# PROCESSED_MASTER_TABLE from elsewhere -- e.g. experiments/scripts/ --
# still resolve it correctly)
PROCESSED_MASTER_TABLE = Path(__file__).resolve().parent / "processed_master_dataset.csv"

# ==========================================================
# Training parameters
# ==========================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

PIN_MEMORY = False

SHUFFLE_TRAIN = True

SHUFFLE_VALIDATION = False

SHUFFLE_TEST = False


def create_dataloaders():

    """
    Creates the train, validation and test DataLoaders.
    """

    df = pd.read_csv(PROCESSED_MASTER_TABLE)

    train_df = df[df["Split"] == "train"].reset_index(drop=True)

    validation_df = df[df["Split"] == "val"].reset_index(drop=True)

    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    train_dataset = ODELIADataset(
        train_df,
        root=PROCESSED_ROOT,
        transform=train_augmentations()
    )

    validation_dataset = ODELIADataset(
        validation_df,
        root=PROCESSED_ROOT
    )

    test_dataset = ODELIADataset(
        test_df,
        root=PROCESSED_ROOT
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_VALIDATION,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TEST,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    class_weights = compute_weights(PROCESSED_MASTER_TABLE)

    return (
        train_loader,
        validation_loader,
        test_loader,
        class_weights
    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader, weights = create_dataloaders()

    print("=" * 60)
    print("Dataset successfully loaded.")
    print("=" * 60)

    print(f"Training patients   : {len(train_loader.dataset)}")
    print(f"Validation patients : {len(val_loader.dataset)}")
    print(f"Test patients       : {len(test_loader.dataset)}")

    print()

    print("Class weights:")
    print(weights)

#########
# Usage #
#########
#from load_dataset import create_dataloaders
#train_loader, val_loader, test_loader, class_weights = create_dataloaders()

#################
# Visualization
#################

# Inspect a random batch of patients
#
# from visualization import show_batch
#
# batch = next(iter(train_loader))
# show_batch(batch, num_patients=4)
#
# The function displays:
# - A random subset of patients from the batch
# - One random MRI slice (same slice for all displayed patients)
# - The three modalities (T2 / Pre / Post1)
# - Patient UID
# - Ground-truth lesion label