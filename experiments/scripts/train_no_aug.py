"""
train_no_aug.py

Clean single-variable ablation against the reference configuration:
identical config (layer4-only unfrozen ResNet18, plain attention pooling,
lr=1e-5, weight_decay=1e-4, batch 16, dropout 0.5, 15 epochs, T2/Pre/Post1)
with only train-time augmentation removed.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

# so root-level modules (dataset.py, model.py, etc.) and data/output paths
# resolve regardless of the working directory this script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from dataset import ODELIADataset
from compute_class_weights import compute_weights
from load_dataset import PROCESSED_ROOT, PROCESSED_MASTER_TABLE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
from model import BreastMRINetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

run_start_time = time.time()

df = pd.read_csv(PROCESSED_MASTER_TABLE)

train_df = df[df["Split"] == "train"].reset_index(drop=True)
val_df = df[df["Split"] == "val"].reset_index(drop=True)
test_df = df[df["Split"] == "test"].reset_index(drop=True)

train_dataset = ODELIADataset(train_df, root=PROCESSED_ROOT, transform=None)
val_dataset = ODELIADataset(val_df, root=PROCESSED_ROOT)
test_dataset = ODELIADataset(test_df, root=PROCESSED_ROOT)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

class_weights = compute_weights(PROCESSED_MASTER_TABLE)
class_weights = class_weights.to(device)

# Model

model = BreastMRINetwork().to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

# single learning rate for all trainable parameters
trainable_params = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(
    trainable_params,
    lr=1e-5,
    weight_decay=1e-4
)

NUM_EPOCHS = 15

best_val_loss = float("inf")

RUN_NAME = "no_aug"

CHECKPOINT_PATH = str(CHECKPOINTS_DIR / f"best_model_{RUN_NAME}.pt")
LOSS_CURVE_PATH = str(FIGURES_DIR / f"loss_curve_{RUN_NAME}.png")
CONFUSION_MATRIX_PATH = str(FIGURES_DIR / f"confusion_matrix_{RUN_NAME}.png")
ROC_CURVE_PATH = str(FIGURES_DIR / f"roc_curve_{RUN_NAME}.png")

def run_epoch(loader, train):

    model.train(train)

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.set_grad_enabled(train):

        for batch in tqdm(loader, desc="Train" if train else "Validate"):

            images = batch["images"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.detach().cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, all_labels, all_predictions, all_probabilities

# Training

train_loss_history = []
val_loss_history = []

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_acc, _, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_acc, _, _, _ = run_epoch(val_loader, train=False)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Saved new best model (val loss {val_loss:.4f})")


# Loss curve

plt.figure(figsize=(8, 5))

plt.plot(range(1, NUM_EPOCHS + 1), train_loss_history, label="Train loss")
plt.plot(range(1, NUM_EPOCHS + 1), val_loss_history, label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss per epoch ({RUN_NAME})")
plt.legend()
plt.tight_layout()

plt.savefig(LOSS_CURVE_PATH)
plt.show()

# Test evaluation (best checkpoint)

model.load_state_dict(torch.load(CHECKPOINT_PATH))

test_loss, test_acc, test_labels, test_predictions, test_probabilities = run_epoch(test_loader, train=False)

print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# Precision / recall / confusion matrix

CLASS_NAMES = ["No lesion", "Benign", "Malignant"]

print()
print(classification_report(
    test_labels,
    test_predictions,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES
))

cm = confusion_matrix(test_labels, test_predictions, labels=[0, 1, 2])

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
).plot()

plt.title(f"Confusion matrix (test set) ({RUN_NAME})")
plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_PATH)
plt.show()

# Sensitivity / specificity / AUC (one-vs-rest)

test_probabilities = np.array(test_probabilities)
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

# ROC curve plot

plt.figure(figsize=(7, 6))

for i, name in enumerate(CLASS_NAMES):

    fpr, tpr, _ = roc_curve(test_labels_bin[:, i], test_probabilities[:, i])
    class_auc = roc_auc_score(test_labels_bin[:, i], test_probabilities[:, i])

    plt.plot(fpr, tpr, label=f"{name} (AUC = {class_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curves, one-vs-rest (test set) ({RUN_NAME})")
plt.legend()
plt.tight_layout()

plt.savefig(ROC_CURVE_PATH)
plt.show()

# Run time

elapsed_seconds = time.time() - run_start_time
elapsed_minutes, elapsed_seconds = divmod(int(elapsed_seconds), 60)
elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)

print(f"Run time ({RUN_NAME}): {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")
