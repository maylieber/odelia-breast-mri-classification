import random
import matplotlib.pyplot as plt
import random
import pandas as pd
import nibabel as nib
from pathlib import Path

LABELS = {0: "No lesion", 1: "Benign", 2: "Malignant"}

def show_batch(batch, num_patients=4):
    """
    Display random patients from a batch.

    Parameters
    ----------
    batch : dict
        Batch returned by the DataLoader.

    num_patients : int
        Number of patients to display.
    """

    images = batch["images"]
    labels = batch["label"]
    uids = batch["uid"]

    num_patients = min(num_patients, len(images))

    chosen = random.sample(range(len(images)), num_patients)

    slice_idx = random.randint(0, images.shape[1] - 1)

    fig, axes = plt.subplots(
        num_patients,
        3,
        figsize=(10, 3 * num_patients)
    )

    if num_patients == 1:
        axes = [axes]

    titles = ["T2", "Pre", "Post1"]

    for row, patient in enumerate(chosen):

        for col in range(3):

            ax = axes[row][col]

            ax.imshow(
                images[patient, slice_idx, col].numpy(),
                cmap="gray"
            )

            if row == 0:
                ax.set_title(titles[col])

            ax.axis("off")

        axes[row][0].set_ylabel(
            f"{uids[patient]}\n"
            f"{LABELS[int(labels[patient])]}",
            fontsize=10
        )

    plt.suptitle(f"Batch inspection (Slice {slice_idx})")

    plt.tight_layout()

    plt.show()

def plot_class_distribution(df_path):

    df = pd.read_csv(df_path)
    splits = ["train", "val", "test"]

    fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
    )

    for ax, split in zip(axes, splits):
        subset = df[df["Split"] == split]

        counts = subset["Lesion"].value_counts().sort_index()

        ax.bar(
            [LABELS[i] for i in counts.index],
            counts.values
        )

        ax.set_title(split.capitalize())

        ax.set_ylabel("Patients")

        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()

    plt.show()

def look_at_random_processed_patient_slices(data_path):
    root = Path(data_path)

    patients = []

    for institution in root.iterdir():

        data_dir = institution

        if data_dir.exists():

            for patient in data_dir.iterdir():
                patients.append(patient)

    patient = random.choice(patients)

    print("Selected patient:")
    print(patient)

    t2 = nib.load(patient / "T2.nii.gz").get_fdata()
    pre = nib.load(patient / "Pre.nii.gz").get_fdata()
    post = nib.load(patient / "post1.nii.gz").get_fdata()

    print("T2 shape:", t2.shape)
    print("Pre shape:", pre.shape)
    print("Sub shape:", post.shape)

    slice_idx = t2.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(t2[:, :, slice_idx], cmap="gray")
    ax[0].set_title("T2")

    ax[1].imshow(pre[:, :, slice_idx], cmap="gray")
    ax[1].set_title("Pre")

    ax[2].imshow(post[:, :, slice_idx], cmap="gray")
    ax[2].set_title("post_1")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()

def look_at_random_patient_slices(data_path):
    root = Path(data_path)

    patients = []

    for institution in root.iterdir():

        data_dir = institution  / "data_unilateral"

        if data_dir.exists():

            for patient in data_dir.iterdir():
                patients.append(patient)

    patient = random.choice(patients)

    print("Selected patient:")
    print(patient)

    t2 = nib.load(patient / "T2.nii.gz").get_fdata()
    pre = nib.load(patient / "Pre.nii.gz").get_fdata()
    post = nib.load(patient / "post_1.nii.gz").get_fdata()

    print("T2 shape:", t2.shape)
    print("Pre shape:", pre.shape)
    print("Sub shape:", post.shape)

    slice_idx = t2.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(t2[:, :, slice_idx], cmap="gray")
    ax[0].set_title("T2")

    ax[1].imshow(pre[:, :, slice_idx], cmap="gray")
    ax[1].set_title("Pre")

    ax[2].imshow(post[:, :, slice_idx], cmap="gray")
    ax[2].set_title("post_1")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


def all_t2_slices(table_path):

    # ==========================================================
    # Load master table
    # ==========================================================

    master_df = pd.read_csv(table_path)

    # ==========================================================
    # Choose random patient
    # ==========================================================

    row = master_df.sample(1).iloc[0]

    print("=" * 60)
    print("Random Patient")
    print("=" * 60)
    print(f"UID          : {row['UID']}")
    print(f"Institution  : {row['Institution']}")
    print(f"Lesion       : {row['Lesion']}")

    # ==========================================================
    # Load T2 volume
    # ==========================================================

    t2 = nib.load(row["T2Path"]).get_fdata()

    print(f"Volume shape: {t2.shape}")

    # ==========================================================
    # Display all slices
    # ==========================================================

    num_slices = t2.shape[2]

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))

    axes = axes.flatten()

    for i in range(num_slices):

        axes[i].imshow(t2[:, :, i], cmap="gray")

        axes[i].set_title(f"Slice {i}", fontsize=9)

        axes[i].axis("off")

    plt.suptitle(
        f"T2 Volume\n"
        f"UID: {row['UID']} | "
        f"{row['Institution']} | "
        f"Lesion: {row['Lesion']}",
        fontsize=14
    )

    plt.tight_layout()

    plt.show()

def show_patient(master_df, uid, dataset_root, slice_idx=None):
    """
    Display one patient from either the original or processed dataset.

    Parameters
    ----------
    master_df : pandas.DataFrame

    uid : str

    dataset_root : str or Path
        Example:
            D:\\ODELIA
            D:\\ODELIA_processed

    slice_idx : int or None
        Slice to display.
        If None, a random slice is chosen.
    """

    dataset_root = Path(dataset_root)

    # ------------------------------------------------------
    # Find patient
    # ------------------------------------------------------

    patient = master_df.loc[master_df["UID"] == uid]

    if len(patient) == 0:
        raise ValueError(f"Patient {uid} not found.")

    patient = patient.iloc[0]

    institution = patient["Institution"]

    # ------------------------------------------------------
    # Build paths
    # ------------------------------------------------------

    patient_folder = dataset_root / institution / uid

    t2_path = patient_folder / "T2.nii.gz"
    pre_path = patient_folder / "Pre.nii.gz"
    post1_path = patient_folder / "Post1.nii.gz"

    # ------------------------------------------------------
    # Load images
    # ------------------------------------------------------

    t2 = nib.load(t2_path).get_fdata()

    pre = nib.load(pre_path).get_fdata()

    post1 = nib.load(post1_path).get_fdata()

    # ------------------------------------------------------
    # Choose slice
    # ------------------------------------------------------

    if slice_idx is None:
        slice_idx = random.randint(0, t2.shape[2] - 1)

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [t2, pre, post1]
    titles = ["T2", "Pre", "Post1"]

    for ax, img, title in zip(axes, images, titles):

        ax.imshow(img[:, :, slice_idx], cmap="gray")

        ax.set_title(title)

        ax.axis("off")

    plt.suptitle(
        f"UID: {uid}\n"
        f"Institution: {institution}    "
        f"Lesion: {patient['Lesion']}    "
        f"Slice: {slice_idx}/{t2.shape[2]-1}",
        fontsize=13
    )

    plt.tight_layout()

    plt.show()