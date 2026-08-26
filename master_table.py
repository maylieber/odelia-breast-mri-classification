import pandas as pd
from pathlib import Path


def create_master_table(data_path, MASTER_TABLE):
    """
    Build a CSV master table for the raw ODELIA dataset: one row per patient,
    with paths to each modality's NIfTI file and a flag for whether each
    file actually exists on disk.

    Parameters
    ----------
    data_path : str or Path
        Root folder of the raw dataset, containing one subfolder per
        institution.

    MASTER_TABLE : str or Path
        Output CSV path.
    """

    dataset_root = Path(data_path)

    output_csv = MASTER_TABLE

    rows = []

    for institution_dir in sorted(dataset_root.iterdir()):

        if not institution_dir.is_dir():
            continue

        institution = institution_dir.name

        metadata_dir = institution_dir / "metadata_unilateral"
        data_dir = institution_dir / "data_unilateral"

        annotation_path = metadata_dir / "annotation.csv"
        split_path = metadata_dir / "split.csv"

        annotation_df = pd.read_csv(annotation_path)
        split_df = pd.read_csv(split_path)

        df = annotation_df.merge(split_df, on="UID")

        for _, row in df.iterrows():

            uid = row["UID"]

            patient_dir = data_dir / uid

            t2_path = patient_dir / "T2.nii.gz"
            pre_path = patient_dir / "Pre.nii.gz"
            post1_path = patient_dir / "Post_1.nii.gz"

            rows.append({

                "UID": uid,

                "Institution": institution,

                "Split": row["Split"],

                "Fold": row["Fold"],

                "Lesion": row["Lesion"],

                "PatientFolder": str(patient_dir),

                "T2Path": str(t2_path),

                "PrePath": str(pre_path),

                "Post1Path": str(post1_path),

                "HasT2": t2_path.exists(),

                "HasPre": pre_path.exists(),

                "HasPost1": post1_path.exists()

            })

    master_df = pd.DataFrame(rows)

    master_df.to_csv(output_csv, index=False)

    print("=" * 50)
    print("Master table created!")
    print("=" * 50)

    print(f"Total samples : {len(master_df)}")

    print("\nColumns:")
    print(master_df.columns.tolist())

    print("\nRandom samples:")
    print(master_df.sample(10))

    print("\nMissing files:")

    print(master_df[
        (~master_df["HasT2"])
        | (~master_df["HasPre"])
        | (~master_df["HasPost1"])
    ])

    print("\nSaved to:")
    print(output_csv.resolve())
