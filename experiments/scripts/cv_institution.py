"""
cv_institution.py

5-fold cross-validation of the reference configuration (layer4-only unfrozen ResNet18,
plain attention pooling, lr=1e-5, weight_decay=1e-4, batch 16, dropout 0.5,
augmentation on, 15 epochs), using INSTITUTION as the fold partition
(leave-one-institution-out per fold).

Fold groups (6 institutions merged down to 5 folds; RUMC is too small
(31 patients, 2 Benign rows) to stand alone, merged into MHA):

    fold1: CAM
    fold2: UMCU
    fold3: MHA + RUMC
    fold4: RSH
    fold5: UKA

For each fold, that fold's institution(s) are held out entirely as the
test set. The remaining institutions' rows are split into train/val
(80/20, patient-grouped so a patient's left/right pair stays together,
stratified by Lesion) via StratifiedGroupKFold.

Outputs, per fold: best_model_cv_inst_fold{N}_{NAMES}.pt,
loss_curve_..., confusion_matrix_..., roc_curve_...
Plus an aggregate cv_institution_summary.csv across all 5 folds, reporting
macro/weighted/micro AUC per fold (micro AUC being the metric BCN-AIM/the
official ODELIA challenge leaderboard actually uses).
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm

# so root-level modules (dataset.py, model.py, etc.) and data/output paths
# resolve regardless of the working directory this script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import ODELIADataset
from augmentations import train_augmentations
from model import BreastMRINetwork
from load_dataset import PROCESSED_ROOT, PROCESSED_MASTER_TABLE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Fold definition
# --------------------------------------------------

FOLD_GROUPS = [
    ("fold1_CAM", ["CAM"]),
    ("fold2_UMCU", ["UMCU"]),
    ("fold3_MHA_RUMC", ["MHA", "RUMC"]),
    ("fold4_RSH", ["RSH"]),
    ("fold5_UKA", ["UKA"]),
]

NUM_EPOCHS = 15  # matches the reference configuration
CLASS_NAMES = ["No lesion", "Benign", "Malignant"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --------------------------------------------------
# Load full processed dataset, add patient-level base_id
# (strips the _left / _right suffix so a patient's two rows group together)
# --------------------------------------------------

full_df = pd.read_csv(PROCESSED_MASTER_TABLE)
full_df["base_id"] = full_df["UID"].str.replace(r"_(left|right)$", "", regex=True)


def run_epoch(loader, train, model, optimizer, criterion):

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


def build_loader(df, augment):

    dataset = ODELIADataset(
        df.reset_index(drop=True),
        root=PROCESSED_ROOT,
        transform=train_augmentations() if augment else None
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=augment,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )


fold_summaries = []

cv_start_time = time.time()

for fold_name, institutions in FOLD_GROUPS:

    print("=" * 60)
    print(f"{fold_name}  (held-out test institution(s): {institutions})")
    print("=" * 60)

    fold_start_time = time.time()

    # ----------------------------------------------
    # Split: held-out institution(s) = test, rest = train/val pool
    # ----------------------------------------------

    test_df = full_df[full_df["Institution"].isin(institutions)].reset_index(drop=True)
    pool_df = full_df[~full_df["Institution"].isin(institutions)].reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    train_idx, val_idx = next(splitter.split(
        pool_df,
        y=pool_df["Lesion"],
        groups=pool_df["base_id"]
    ))

    train_df = pool_df.iloc[train_idx].reset_index(drop=True)
    val_df = pool_df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

    # ----------------------------------------------
    # Loaders + class weights (recomputed per fold, from this fold's train split)
    # ----------------------------------------------

    train_loader = build_loader(train_df, augment=True)
    val_loader = build_loader(val_df, augment=False)
    test_loader = build_loader(test_df, augment=False)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["Lesion"].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ----------------------------------------------
    # Fresh model per fold (reference configuration: layer4-only unfrozen, plain attention)
    # ----------------------------------------------

    model = BreastMRINetwork().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=1e-5,
        weight_decay=1e-4
    )

    CHECKPOINT_PATH = str(CHECKPOINTS_DIR / f"best_model_cv_inst_{fold_name}.pt")
    LOSS_CURVE_PATH = str(FIGURES_DIR / f"loss_curve_cv_inst_{fold_name}.png")
    CONFUSION_MATRIX_PATH = str(FIGURES_DIR / f"confusion_matrix_cv_inst_{fold_name}.png")
    ROC_CURVE_PATH = str(FIGURES_DIR / f"roc_curve_cv_inst_{fold_name}.png")

    best_val_loss = float("inf")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_acc, _, _, _ = run_epoch(train_loader, True, model, optimizer, criterion)
        val_loss, val_acc, _, _, _ = run_epoch(val_loader, False, model, optimizer, criterion)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(
            f"[{fold_name}] Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"[{fold_name}] Saved new best model (val loss {val_loss:.4f})")

    # ----------------------------------------------
    # Loss curve
    # ----------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_loss_history, label="Train loss")
    plt.plot(range(1, NUM_EPOCHS + 1), val_loss_history, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per epoch ({fold_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()

    # ----------------------------------------------
    # Test evaluation (best checkpoint)
    # ----------------------------------------------

    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    test_loss, test_acc, test_labels, test_predictions, test_probabilities = run_epoch(
        test_loader, False, model, optimizer, criterion
    )

    print(f"[{fold_name}] Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    report_str = classification_report(
        test_labels, test_predictions, labels=[0, 1, 2], target_names=CLASS_NAMES
    )
    print(report_str)

    cm = confusion_matrix(test_labels, test_predictions, labels=[0, 1, 2])

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot()
    plt.title(f"Confusion matrix (test set) ({fold_name})")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    test_probabilities = np.array(test_probabilities)
    test_labels_bin = label_binarize(test_labels, classes=[0, 1, 2])

    per_class = {}

    for i, name in enumerate(CLASS_NAMES):

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        # AUC undefined if the test fold has only one class present for this label
        if len(np.unique(test_labels_bin[:, i])) > 1:
            auc = roc_auc_score(test_labels_bin[:, i], test_probabilities[:, i])
        else:
            auc = float("nan")

        per_class[name] = {"sensitivity": sensitivity, "specificity": specificity, "auc": auc}

        print(f"{name:<12} Sens {sensitivity:.4f}  Spec {specificity:.4f}  AUC {auc}")

    try:
        macro_auc = roc_auc_score(test_labels_bin, test_probabilities, average="macro")
        weighted_auc = roc_auc_score(test_labels_bin, test_probabilities, average="weighted")
        micro_auc = roc_auc_score(test_labels_bin, test_probabilities, average="micro")
    except ValueError:
        macro_auc = float("nan")
        weighted_auc = float("nan")
        micro_auc = float("nan")

    print(f"[{fold_name}] Macro-average AUC: {macro_auc:.4f} | Weighted-average AUC: {weighted_auc:.4f} | Micro-average AUC: {micro_auc:.4f}")

    # ----------------------------------------------
    # ROC curve
    # ----------------------------------------------

    plt.figure(figsize=(7, 6))
    for i, name in enumerate(CLASS_NAMES):
        if len(np.unique(test_labels_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(test_labels_bin[:, i], test_probabilities[:, i])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {per_class[name]['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves, one-vs-rest (test set) ({fold_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH)
    plt.close()

    fold_elapsed = time.time() - fold_start_time

    fold_summaries.append({
        "fold": fold_name,
        "institutions": "+".join(institutions),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "test_acc": test_acc,
        "test_loss": test_loss,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "micro_auc": micro_auc,
        "no_lesion_sens": per_class["No lesion"]["sensitivity"],
        "no_lesion_spec": per_class["No lesion"]["specificity"],
        "no_lesion_auc": per_class["No lesion"]["auc"],
        "benign_sens": per_class["Benign"]["sensitivity"],
        "benign_spec": per_class["Benign"]["specificity"],
        "benign_auc": per_class["Benign"]["auc"],
        "malignant_sens": per_class["Malignant"]["sensitivity"],
        "malignant_spec": per_class["Malignant"]["specificity"],
        "malignant_auc": per_class["Malignant"]["auc"],
        "run_time_seconds": fold_elapsed,
    })

    print(f"[{fold_name}] Fold time: {fold_elapsed / 60:.1f} min")

# --------------------------------------------------
# Aggregate summary across folds
# --------------------------------------------------

summary_df = pd.DataFrame(fold_summaries)
summary_df.to_csv(CHECKPOINTS_DIR / "cv_institution_summary.csv", index=False)

print("=" * 60)
print("5-fold institution-based CV summary (reference configuration)")
print("=" * 60)
print(summary_df.to_string(index=False))

numeric_cols = [
    "test_acc", "test_loss", "macro_auc", "weighted_auc", "micro_auc",
    "no_lesion_sens", "no_lesion_spec", "no_lesion_auc",
    "benign_sens", "benign_spec", "benign_auc",
    "malignant_sens", "malignant_spec", "malignant_auc",
]

print()
print("Mean +/- std across folds:")
for col in numeric_cols:
    print(f"{col:<16} {summary_df[col].mean():.4f} +/- {summary_df[col].std():.4f}")

total_elapsed = time.time() - cv_start_time
hours, remainder = divmod(int(total_elapsed), 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nTotal CV run time: {hours}h {minutes}m {seconds}s")