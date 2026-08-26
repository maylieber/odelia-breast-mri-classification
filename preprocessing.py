import nibabel as nib
import numpy as np
import torchio as tio
from roi import compute_roi
import pandas as pd
from pathlib import Path


def process_patient(
        row,
        output_root,
        output_size=(128, 128, 32),
        percentile=60,
        margin=5,
):
    """
    Preprocess a single patient.

    Steps
    -----
    1. Load T2 / Pre / Post1
    2. Compute ROI from T2
    3. Crop all modalities
    4. Resize to output_size
    5. Save processed images

    Parameters
    ----------
    row : pandas.Series
        One row from master_df.

    output_root : str or Path
        Root folder of processed dataset.

    output_size : tuple
        Desired output size (H, W, D).

    percentile : int
        ROI threshold percentile.

    margin : int
        ROI margin.

    Returns
    -------
    Path
        Folder where the processed patient was saved.
    """

    output_root = Path(output_root)
    uid = row["UID"]
    institution = row["Institution"]

    patient_folder = output_root / institution / uid
    patient_folder.mkdir(parents=True, exist_ok=True)

    t2 = nib.load(row["T2Path"]).get_fdata()

    pre = nib.load(row["PrePath"]).get_fdata()

    post1 = nib.load(row["Post1Path"]).get_fdata()

    xmin, xmax, ymin, ymax = compute_roi(
        t2,
        percentile=percentile,
        margin=margin
    )

    t2 = t2[xmin:xmax + 1,
            ymin:ymax + 1,
            :]

    pre = pre[xmin:xmax + 1,
              ymin:ymax + 1,
              :]

    post1 = post1[xmin:xmax + 1,
                  ymin:ymax + 1,
                  :]

    def resize_volume(volume):

        image = tio.ScalarImage(
            tensor=volume[np.newaxis].astype(np.float32)
        )

        resize = tio.Resize(output_size)

        image = resize(image)

        return image

    t2 = resize_volume(t2)

    pre = resize_volume(pre)

    post1 = resize_volume(post1)

    t2.save(patient_folder / "T2.nii.gz")

    pre.save(patient_folder / "Pre.nii.gz")

    post1.save(patient_folder / "Post1.nii.gz")

    return {
        "patient_folder": patient_folder,
        "crop_shape": t2.shape[1:],  # after resize
        "roi": (xmin, xmax, ymin, ymax)
    }


def preprocess_dataset(
        master_table_path,
        output_root,
        processed_master_table_path,
        output_size=(128, 128, 32),
        percentile=60,
        margin=5,
):
    """
    Preprocess the entire ODELIA dataset.

    Parameters
    ----------
    master_table_path : str or Path
        Path to the master table.

    output_root : str or Path
        Root folder for processed images.

    processed_master_table_path : str or Path
        Output CSV containing processed dataset information.

    Returns
    -------
    pandas.DataFrame
        Processed dataframe.
    """

    master_df = pd.read_csv(master_table_path)

    processed_rows = []

    print("=" * 60)
    print("Preprocessing patients...")
    print("=" * 60)

    total = len(master_df)

    for i, (_, row) in enumerate(master_df.iterrows(), start=1):

        process_patient(
            row=row,
            output_root=output_root,
            output_size=output_size,
            percentile=percentile,
            margin=margin
        )

        patient_folder = Path(output_root) / row["Institution"] / row["UID"]

        t2_path = patient_folder / "T2.nii.gz"
        pre_path = patient_folder / "Pre.nii.gz"
        post1_path = patient_folder / "Post1.nii.gz"

        processed_rows.append({

            "UID": row["UID"],

            "Institution": row["Institution"],

            "Split": row["Split"],

            "Fold": row["Fold"],

            "Lesion": row["Lesion"],

            "PatientFolder": str(patient_folder),

            "T2Path": str(t2_path),

            "PrePath": str(pre_path),

            "Post1Path": str(post1_path),

            "HasT2": t2_path.exists(),

            "HasPre": pre_path.exists(),

            "HasPost1": post1_path.exists()
        })

        if i % 25 == 0 or i == total:
            print(f"{i}/{total} patients processed")

    processed_df = pd.DataFrame(processed_rows)

    processed_df.to_csv(
        processed_master_table_path,
        index=False
    )

    print()
    print("Processed dataset saved.")

    return processed_df
