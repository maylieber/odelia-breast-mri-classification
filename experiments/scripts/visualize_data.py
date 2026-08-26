from pathlib import Path
import random
import nibabel as nib
import matplotlib.pyplot as plt

root = Path(r"D:\ODELIA")

patients = []

for institution in root.iterdir():

    data_dir = institution / "data_unilateral"

    if data_dir.exists():

        for patient in data_dir.iterdir():
            patients.append(patient)

patient = random.choice(patients)

print("Selected patient:")
print(patient)

t2 = nib.load(patient / "T2.nii.gz").get_fdata()
pre = nib.load(patient / "Pre.nii.gz").get_fdata()
sub = nib.load(patient / "Sub_1.nii.gz").get_fdata()
post = nib.load(patient / "Post_1.nii.gz").get_fdata()

print("T2 shape:", t2.shape)
print("Pre shape:", pre.shape)
print("Sub shape:", sub.shape)
print("Post_1 shape:", post.shape)

slice_idx = t2.shape[2] // 2

fig, ax = plt.subplots(1, 4, figsize=(15, 5))

ax[0].imshow(t2[:, :, slice_idx], cmap="gray")
ax[0].set_title("T2")

ax[1].imshow(pre[:, :, slice_idx], cmap="gray")
ax[1].set_title("Pre")

ax[2].imshow(sub[:, :, slice_idx], cmap="gray")
ax[2].set_title("Sub_1")

ax[3].imshow(post[:, :, slice_idx], cmap="gray")
ax[3].set_title("Post_1")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()