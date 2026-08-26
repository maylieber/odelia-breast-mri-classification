import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from load_dataset import create_dataloaders
from model import BreastMRINetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

run_start_time = time.time()


train_loader, val_loader, test_loader, class_weights = create_dataloaders()

class_weights = class_weights.to(device)

# Overfit sanity check
# Set to a small int (e.g. 8) to train/validate/test on the same
# tiny subset of the train set, to confirm the pipeline can memorize
# it. Set back to None for real training.

OVERFIT_SUBSET_SIZE = None
OVERFIT_SEED = 42

if OVERFIT_SUBSET_SIZE is not None:

    random.seed(OVERFIT_SEED)
    subset_indices = random.sample(range(len(train_loader.dataset)), OVERFIT_SUBSET_SIZE)

    overfit_subset = Subset(train_loader.dataset, subset_indices)

    train_loader = DataLoader(overfit_subset, batch_size=OVERFIT_SUBSET_SIZE, shuffle=True)
    val_loader = DataLoader(overfit_subset, batch_size=OVERFIT_SUBSET_SIZE, shuffle=False)
    test_loader = DataLoader(overfit_subset, batch_size=OVERFIT_SUBSET_SIZE, shuffle=False)


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

NUM_EPOCHS = 100 if OVERFIT_SUBSET_SIZE is not None else 15

best_val_loss = float("inf")

RUN_NAME = "run" if OVERFIT_SUBSET_SIZE is None else f"overfit{OVERFIT_SUBSET_SIZE}"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures", exist_ok=True)

CHECKPOINT_PATH = f"checkpoints/best_model_{RUN_NAME}.pt"
LOSS_CURVE_PATH = f"figures/loss_curve_{RUN_NAME}.png"
CONFUSION_MATRIX_PATH = f"figures/confusion_matrix_{RUN_NAME}.png"
ROC_CURVE_PATH = f"figures/roc_curve_{RUN_NAME}.png"

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
micro_auc = roc_auc_score(test_labels_bin, test_probabilities, average="micro")

print(f"\nMacro-average AUC:    {macro_auc:.4f}")
print(f"Weighted-average AUC: {weighted_auc:.4f}")
print(f"Micro-average AUC:    {micro_auc:.4f}")


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
