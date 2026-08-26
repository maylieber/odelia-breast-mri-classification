"""
Parallel dataset wrapper for the two-stage (hierarchical) cascade classification experiment.

Reuses ODELIADataset's loading/ROI-cropping/normalization/augmentation
logic completely unchanged -- the only difference is that __getitem__'s
returned label is remapped through a label_map dict after the parent
class builds it, so the same T2/Pre/Post1 volumes back both stages'
dataloaders. dataset.py itself is untouched.
"""

import sys
from pathlib import Path
import torch

# so root-level modules (dataset.py, etc.) resolve regardless of the
# working directory this script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset import ODELIADataset


class ODELIADatasetRemapped(ODELIADataset):

    def __init__(self, dataframe, root, label_map, transform=None):
        """
        label_map: dict mapping the original 3-way Lesion label
            (0=No lesion, 1=Benign, 2=Malignant) to the binary label this
            stage actually trains on, e.g.
                Stage 1: {0: 0, 1: 1, 2: 1}   (No lesion vs. Lesion)
                Stage 2: {1: 0, 2: 1}         (Benign vs. Malignant --
                    dataframe must already be filtered to Lesion in {1, 2})
        """
        super().__init__(dataframe, root, transform=transform)
        self.label_map = label_map

    def __getitem__(self, idx):

        item = super().__getitem__(idx)

        original_label = int(item["label"].item())
        item["label"] = torch.tensor(self.label_map[original_label], dtype=torch.long)

        return item