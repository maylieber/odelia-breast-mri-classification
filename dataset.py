import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path

class ODELIADataset(Dataset):

    def __init__(self, dataframe, root, transform=None):
        """
        dataframe: filtered master_df (train/val/test)
        root: path to processed dataset
        """
        self.df = dataframe.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_nifti(self, path):
        """Load nifti as numpy array"""
        return nib.load(str(path)).get_fdata()

    def normalize(self, img):
        """Simple min-max normalization"""
        img = img.astype(np.float32)
        img = img - np.min(img)
        if np.max(img) > 0:
            img = img / np.max(img)
        return img

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        uid = row["UID"]
        institution = row["Institution"]
        label = int(row["Lesion"])

        patient_folder = self.root / institution / uid

        t2_path = patient_folder / "T2.nii.gz"
        pre_path = patient_folder / "Pre.nii.gz"
        post1_path = patient_folder / "Post1.nii.gz"

        t2 = self.normalize(self.load_nifti(t2_path))
        pre = self.normalize(self.load_nifti(pre_path))
        post1 = self.normalize(self.load_nifti(post1_path))

        slices = []

        for i in range(32): # num_slices

            t2_slice = t2[:, :, i]
            pre_slice = pre[:, :, i]
            post_slice = post1[:, :, i]

            # stack modalities → (3, H, W)
            img = np.stack([t2_slice, pre_slice, post_slice], axis=0)

            slices.append(img)

        # (32, 3, H, W)
        slices = np.stack(slices, axis=0)

        images = torch.tensor(slices, dtype=torch.float32)

        if self.transform is not None:
            images = self.transform(images)

        label = torch.tensor(label, dtype=torch.long)

        return {
            "images": images,
            "label": label,
            "uid": uid
        }
