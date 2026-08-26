"""
Two-stage cascade, combined evaluation: chains Stage 1 (No lesion vs.
Lesion) and Stage 2 (Benign vs. Malignant) checkpoints back into a single
3-way prediction on the same test set/labels the reference configuration
was evaluated on, so the two-stage cascade can be compared directly
against the reference configuration's numbers.

The POINT PREDICTION follows the actual two-step hard cascade requested:
Stage 1 decides No lesion vs. Lesion first (argmax of its own 2-way
softmax); only if it says Lesion does Stage 2 run, deciding Benign vs.
Malignant (argmax of its own 2-way softmax). This is NOT the same as
taking argmax over chain-rule-combined probabilities -- an earlier
version of this script did that and it silently reclassified genuine
lesion cases back to "No lesion" whenever Stage 2 was a near-toss-up
(since P(Benign|input) and P(Malignant|input) can each individually
fall below P(No lesion) even though their sum, P(Lesion), exceeds it).
That produced a spurious Malignant-recall collapse that was an artifact
of the combination math, not of the cascade itself -- fixed here.

For AUC/ROC (which need a full probability vector, not just a hard
label), the chain rule is still used to build one:

    P(No lesion) = P_stage1(No lesion)
    P(Benign)    = P_stage1(Lesion) * P_stage2(Benign)
    P(Malignant) = P_stage1(Lesion) * P_stage2(Malignant)

This is a legitimate probability distribution (it sums to 1) and ranks
patients consistently with the two-step decision at the P=0.5 boundary
per stage, but it is used ONLY for AUC below, never for the point
prediction/accuracy/confusion-matrix numbers.

Requires best_model_stage1.pt and best_model_stage2.pt to already exist
(train_stage1.py / train_stage2.py). Does not retrain anything.
train.py / model.py / dataset.py are untouched.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

# so root-level modules (dataset.py, etc.) and data/output paths resolve
# regardless of the working directory this script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import ODELIADataset
from model_hierarchical import BreastMRINetworkHierarchical

# --------------------------------------------------
# Settings (match reference configuration / stage1 / stage2)
# --------------------------------------------------

CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_ROOT = Path(r"D:\ODELIA_processed")
PROCESSED_MASTER_TABLE = PROJECT_ROOT / "processed_master_dataset.csv"

RUN_NAME = "cascade"
CLASS_NAMES = ["No lesion", "Benign", "Malignant"]

STAGE1_CHECKPOINT = str(CHECKPOINTS_DIR / "best_model_stage1.pt")
STAGE2_CHECKPOINT = str(CHECKPOINTS_DIR / "best_model_stage2.pt")

CONFUSION_MATRIX_PATH = str(FIGURES_DIR / f"confusion_matrix_{RUN_NAME}.png")
ROC_CURVE_PATH = str(FIGURES_DIR / f"roc_curve_{RUN_NAME}.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

# --------------------------------------------------
# Test set: full 3-way labels, no augmentation, reference configuration's normalization
# --------------------------------------------------

df = pd.read_csv(PROCESSED_MASTER_TABLE)
test_df = df[df["Split"] == "test"].reset_index(drop=True)

test_dataset = ODELIADataset(
    test_df, root=PROCESSED_ROOT, transform=None
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# --------------------------------------------------
# Load both stage checkpoints
# --------------------------------------------------

stage1_model = BreastMRINetworkHierarchical(num_classes=2).to(device)
stage1_model.load_state_dict(torch.load(STAGE1_CHECKPOINT, map_location=device))
stage1_model.eval()

stage2_model = BreastMRINetworkHierarchical(num_classes=2).to(device)
stage2_model.load_state_dict(torch.load(STAGE2_CHECKPOINT, map_location=device))
stage2_model.eval()

# --------------------------------------------------
# Run the cascade over the test set
# --------------------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for batch in tqdm(test_loader, desc="Cascade eval"):

        images = batch["images"].to(device)
        labels = batch["label"].to(device)

        stage1_probs = torch.softmax(stage1_model(images), dim=1)  # (B, 2): [No lesion, Lesion]
        stage2_probs = torch.softmax(stage2_model(images), dim=1)  # (B, 2): [Benign, Malignant]

        p_no_lesion = stage1_probs[:, 0]
        p_lesion = stage1_probs[:, 1]

        # Chain-rule probability vector -- used only for AUC/ROC below.
        p_benign = p_lesion * stage2_probs[:, 0]
        p_malignant = p_lesion * stage2_probs[:, 1]
        combined_probs = torch.stack([p_no_lesion, p_benign, p_malignant], dim=1)  # (B, 3)

        # True two-step hard cascade -- this is the actual point prediction.
        # Stage 1 decides No lesion (0) vs. Lesion first; only if Lesion does
        # Stage 2 decide Benign (1) vs. Malignant (2).
        stage1_decision = stage1_probs.argmax(dim=1)  # 0 = No lesion, 1 = Lesion
        stage2_decision = stage2_probs.argmax(dim=1)  # 0 = Benign, 1 = Malignant

        predictions = torch.where(
            stage1_decision == 0,
            torch.zeros_like(stage1_decision),
            stage2_decision + 1
        )

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(combined_probs.cpu().tolist())

test_labels = all_labels
test_predictions = all_predictions
test_probabilities = np.array(all_probabilities)

test_acc = np.mean(np.array(test_labels) == np.array(test_predictions))
print(f"\nCombined cascade test accuracy: {test_acc:.4f}")

# --------------------------------------------------
# Precision / recall / confusion matrix (same format as train.py)
# --------------------------------------------------

print()
print(classification_report(
    test_labels, test_predictions, labels=[0, 1, 2], target_names=CLASS_NAMES
))

cm = confusion_matrix(test_labels, test_predictions, labels=[0, 1, 2])

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot()
plt.title(f"Confusion matrix (test set) ({RUN_NAME}, hierarchical cascade)")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close()

# --------------------------------------------------
# Sensitivity / specificity / AUC (one-vs-rest) -- same computation as train.py
# --------------------------------------------------

test_labels_bin = label_binarize(test_labels, classes=[0, 1, 2])

print(f"{'Class':<12}{'Sensitivity':<14}{'Specificity':<14}{'AUC':<10}")

for i, name in enumerate(CLASS_NAMES):

    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = roc_auc_score(test_labels_bin[:, i], test_probabilities[:, i])

    print(f"{name:<12}{sensitivity:<14.4f}{specificity:<14.4f}{auc:<10.4f}")

macro_auc = roc_auc_score(test_labels_bin, test_probabilities, average="macro")
weighted_auc = roc_auc_score(test_labels_bin, test_probabilities, average="weighted")

print(f"\nMacro-average AUC:    {macro_auc:.4f}")
print(f"Weighted-average AUC: {weighted_auc:.4f}")

# --------------------------------------------------
# ROC curve plot
# --------------------------------------------------

plt.figure(figsize=(7, 6))

for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(test_labels_bin[:, i], test_probabilities[:, i])
    class_auc = roc_auc_score(test_labels_bin[:, i], test_probabilities[:, i])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {class_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curves, one-vs-rest (test set) ({RUN_NAME}, hierarchical cascade)")
plt.legend()
plt.tight_layout()

plt.savefig(ROC_CURVE_PATH)
plt.close()

print(f"\nSaved {CONFUSION_MATRIX_PATH} and {ROC_CURVE_PATH}")