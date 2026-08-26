import random
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def initial_ROI_cropping_for_random(table_path):
    """
    Load a random patient from the master table and display the T2/Pre/Post1
    slices before and after ROI cropping, for visually inspecting and tuning
    the crop parameters.
    """

    # ==========================================================
    # Settings
    # ==========================================================

    MIDDLE_START = 8
    MIDDLE_END = 24          # Python slice -> slices 8...23

    THRESHOLD_PERCENTILE = 60 # Tunable hyperparameter
    MARGIN = 5

    # ==========================================================
    # Load master table
    # ==========================================================

    master_df = pd.read_csv(table_path)

    row = master_df.sample(1).iloc[0]

    print("=" * 60)
    print("Random Patient")
    print("=" * 60)
    print(f"UID          : {row['UID']}")
    print(f"Institution  : {row['Institution']}")
    print(f"Lesion       : {row['Lesion']}")
    print()

    # ==========================================================
    # Load MRI volumes
    # ==========================================================

    t2 = nib.load(row["T2Path"]).get_fdata()

    pre = nib.load(row["PrePath"]).get_fdata()

    post1 = nib.load(row["Post1Path"]).get_fdata()

    print("Original shape:", t2.shape)

    # ==========================================================
    # Compute MIP from middle 16 slices
    # ==========================================================

    middle_volume = t2[:, :, MIDDLE_START:MIDDLE_END]

    mip = np.max(middle_volume, axis=2)

    # ==========================================================
    # Threshold MIP
    # ==========================================================

    foreground = mip[mip > 0]

    threshold = np.percentile(foreground, THRESHOLD_PERCENTILE)

    mask = mip > threshold

    print(f"Threshold = {threshold:.2f}")

    # ==========================================================
    # Compute 2D bounding box
    # ==========================================================

    coords = np.argwhere(mask)

    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()

    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()

    # ==========================================================
    # Add margin
    # ==========================================================

    xmin = max(0, xmin - MARGIN)
    xmax = min(t2.shape[0] - 1, xmax + MARGIN)

    ymin = max(0, ymin - MARGIN)
    ymax = min(t2.shape[1] - 1, ymax + MARGIN)

    print("\nBounding box")
    print("-------------------------")
    print(f"x : {xmin} -> {xmax}")
    print(f"y : {ymin} -> {ymax}")

    # ==========================================================
    # Crop all modalities
    # ==========================================================

    t2_crop = t2[xmin:xmax + 1, ymin:ymax + 1, :]

    pre_crop = pre[xmin:xmax + 1, ymin:ymax + 1, :]

    post1_crop = post1[xmin:xmax + 1, ymin:ymax + 1, :]

    print("Cropped shape:", t2_crop.shape)

    # ==========================================================
    # Display one random slice
    # ==========================================================

    slice_idx = random.randint(0, t2.shape[2] - 1)

    print(f"Displaying slice {slice_idx}")

    # ==========================================================
    # Plot
    # ==========================================================

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    titles = ["T2", "Pre", "Post1"]

    original_images = [t2, pre, post1]

    cropped_images = [t2_crop, pre_crop, post1_crop]

    # ---------------- Original ----------------

    for ax, img, title in zip(axes[0], original_images, titles):

        ax.imshow(img[:, :, slice_idx], cmap="gray")

        rect = Rectangle(
            (ymin, xmin),
            ymax - ymin,
            xmax - xmin,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.set_title(f"{title} - Original")

        ax.axis("off")

    # ---------------- Cropped ----------------

    for ax, img, title in zip(axes[1], cropped_images, titles):

        ax.imshow(img[:, :, slice_idx], cmap="gray")

        ax.set_title(f"{title} - Cropped")

        ax.axis("off")

    plt.suptitle(
        f"{row['UID']} | {row['Institution']} | Lesion: {row['Lesion']}",
        fontsize=15
    )

    plt.tight_layout()

    plt.show()

def compute_roi(
        t2,
        middle_start=8,
        middle_end=24,
        percentile=60,
        margin=5,
):
    """
    Compute a 2D ROI from the T2 volume.

    Steps:
    1. Use the middle slices only.
    2. Compute a Maximum Intensity Projection (MIP).
    3. Threshold the MIP.
    4. Compute a 2D bounding box.
    5. Add a safety margin.

    Parameters
    ----------
    t2 : np.ndarray
        T2 MRI volume with shape (H, W, D).

    middle_start : int
        First middle slice (inclusive).

    middle_end : int
        Last middle slice (exclusive).

    percentile : float
        Percentile used for thresholding.

    margin : int
        Margin added to each side of the ROI.

    Returns
    -------
    xmin, xmax, ymin, ymax : int
        Bounding box coordinates.
    """

    # ------------------------------------------------------
    # Middle slices
    # ------------------------------------------------------

    middle_volume = t2[:, :, middle_start:middle_end]

    # ------------------------------------------------------
    # Maximum Intensity Projection
    # ------------------------------------------------------

    mip = np.max(middle_volume, axis=2)

    # ------------------------------------------------------
    # Threshold
    # ------------------------------------------------------

    foreground = mip[mip > 0]

    if len(foreground) == 0:
        raise ValueError("No foreground voxels found.")

    threshold = np.percentile(foreground, percentile)

    mask = mip > threshold

    # ------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------

    coords = np.argwhere(mask)

    if len(coords) == 0:
        raise ValueError("ROI mask is empty.")

    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()

    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()

    # ------------------------------------------------------
    # Margin
    # ------------------------------------------------------

    xmin = max(0, xmin - margin)
    xmax = min(t2.shape[0] - 1, xmax + margin)

    ymin = max(0, ymin - margin)
    ymax = min(t2.shape[1] - 1, ymax + margin)

    return xmin, xmax, ymin, ymax