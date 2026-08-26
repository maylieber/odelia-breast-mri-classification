import random
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_ROOT = Path(r"D:\ODELIA_processed")
PROCESSED_MASTER_TABLE = PROJECT_ROOT / "processed_master_dataset.csv"
CLASS_NAMES = ["No lesion", "Benign", "Malignant"]
PATIENTS_PER_CLASS = 2

df = pd.read_csv(PROCESSED_MASTER_TABLE)

random.seed(0)

rows = []

for label in [0, 1, 2]:
    subset = df[df["Lesion"] == label]
    sample = subset.sample(PATIENTS_PER_CLASS, random_state=0)
    rows.extend(sample.to_dict("records"))

fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))

titles = ["T2", "Pre", "Post1"]

for i, row in enumerate(rows):

    patient_folder = PROCESSED_ROOT / row["Institution"] / row["UID"]

    t2 = nib.load(patient_folder / "T2.nii.gz").get_fdata()
    pre = nib.load(patient_folder / "Pre.nii.gz").get_fdata()
    post1 = nib.load(patient_folder / "Post1.nii.gz").get_fdata()

    slice_idx = t2.shape[2] // 2

    volumes = [t2, pre, post1]

    for j, (volume, title) in enumerate(zip(volumes, titles)):

        ax = axes[i, j]

        ax.imshow(volume[:, :, slice_idx], cmap="gray")

        if j == 0:
            ax.set_title(f"{title}\n{row['UID']} | {CLASS_NAMES[int(row['Lesion'])]}", fontsize=9)
        else:
            ax.set_title(title)

        ax.axis("off")

plt.suptitle("Processed sample check (middle slice, shape 128x128x32)")
plt.tight_layout()

output_path = FIGURES_DIR / "processed_samples_check.png"
plt.savefig(output_path)
print(f"Saved to {output_path}")