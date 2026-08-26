"""
setup_dataset.py

Run this script once after downloading the ODELIA dataset.

It will:

1. Create the master table
2. Crop every patient
3. Resize every image
4. Save the processed dataset
5. Create the processed master table
"""

from pathlib import Path
from master_table import create_master_table
from preprocessing import preprocess_dataset

# USER SETTINGS
# Original dataset folder path
ORIGINAL_DATASET = Path(r"D:\ODELIA")
# Processed dataset folder path
PROCESSED_DATASET = Path(r"D:\ODELIA_processed")
# Master table output path
MASTER_TABLE = Path(r"master_dataset.csv")
# Processed master table output path
PROCESSED_MASTER_TABLE = Path(r"processed_master_dataset.csv")

# PREPROCESSING PARAMETERS

OUTPUT_SIZE = (128, 128, 32)

ROI_PERCENTILE = 60

ROI_MARGIN = 5

# Create master table

print("=" * 60)
print("STEP 1 / 2")
print("Creating master table")
print("=" * 60)

create_master_table(ORIGINAL_DATASET, MASTER_TABLE)

print()

# Preprocess dataset

print("=" * 60)
print("STEP 2 / 2")
print("Cropping and resizing all patients")
print("=" * 60)

preprocess_dataset(
    master_table_path=MASTER_TABLE,
    output_root=PROCESSED_DATASET,
    processed_master_table_path=PROCESSED_MASTER_TABLE,
    output_size=OUTPUT_SIZE,
    percentile=ROI_PERCENTILE,
    margin=ROI_MARGIN
)

print()
print("=" * 60)
print("Dataset setup completed successfully.")
print("=" * 60)


# Optional visual sanity checks (uncomment the call to run one manually)

from visualization import plot_class_distribution
# plot_class_distribution(PROCESSED_MASTER_TABLE)               # class balance per split

from visualization import look_at_random_patient_slices
# look_at_random_patient_slices(ORIGINAL_DATASET)                # random raw patient, all 3 modalities

from visualization import look_at_random_processed_patient_slices
# look_at_random_processed_patient_slices(PROCESSED_DATASET)     # random processed patient, all 3 modalities

from visualization import all_t2_slices
# all_t2_slices(PROCESSED_MASTER_TABLE)                          # every T2 slice of one random patient

from visualization import show_patient
# show_patient(master_df, uid, dataset_root, slice_idx=None)     # one specific patient/slice
