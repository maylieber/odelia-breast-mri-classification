"""
inference.py

Run the trained reference-configuration model (checkpoints/best_model_run.pt)
without retraining. Two modes:

1. Single patient, already preprocessed the same way as setup_dataset.py
   (ROI-cropped, resized to 128x128x32; folder contains T2.nii.gz,
   Pre.nii.gz, Post1.nii.gz):

       python inference.py --patient_dir path/to/patient_folder

2. Full processed test split, reproducing the reference configuration's
   test-set numbers (accuracy 55.3%, macro AUC 0.689):

       python inference.py --test_set
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

from model import BreastMRINetwork
from dataset import ODELIADataset
from load_dataset import create_dataloaders

CHECKPOINT_PATH = "checkpoints/best_model_run.pt"
CLASS_NAMES = ["No lesion", "Benign", "Malignant"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reuses ODELIADataset's own volume loading/normalization (load_nifti,
# normalize) so single-patient inference can never drift from how the
# training/test dataloaders preprocess the same files.
_volume_loader = ODELIADataset(dataframe=pd.DataFrame(), root=".")


def load_model(checkpoint_path=CHECKPOINT_PATH):

    model = BreastMRINetwork().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def load_patient_volume(patient_dir):
    """
    Loads T2/Pre/Post1 from a preprocessed patient folder and returns a
    (1, 32, 3, 128, 128) tensor ready for BreastMRINetwork.
    """

    patient_dir = Path(patient_dir)

    t2 = _volume_loader.normalize(_volume_loader.load_nifti(patient_dir / "T2.nii.gz"))
    pre = _volume_loader.normalize(_volume_loader.load_nifti(patient_dir / "Pre.nii.gz"))
    post1 = _volume_loader.normalize(_volume_loader.load_nifti(patient_dir / "Post1.nii.gz"))

    slices = []

    for i in range(32):
        img = np.stack([t2[:, :, i], pre[:, :, i], post1[:, :, i]], axis=0)
        slices.append(img)

    slices = np.stack(slices, axis=0)

    images = torch.tensor(slices, dtype=torch.float32).unsqueeze(0)

    return images


def predict_patient(model, patient_dir):

    images = load_patient_volume(patient_dir).to(device)

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_class = int(probabilities.argmax())

    return predicted_class, probabilities


def evaluate_test_set(model):

    _, _, test_loader, _ = create_dataloaders()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["images"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)

    print(f"Test accuracy: {accuracy:.4f}\n")

    print(classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES
    ))

    all_probabilities = np.array(all_probabilities)
    labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])

    print(f"{'Class':<12}{'Sensitivity':<14}{'Specificity':<14}{'AUC':<10}")

    for i, name in enumerate(CLASS_NAMES):

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        auc = roc_auc_score(labels_bin[:, i], all_probabilities[:, i])

        print(f"{name:<12}{sensitivity:<14.4f}{specificity:<14.4f}{auc:<10.4f}")

    macro_auc = roc_auc_score(labels_bin, all_probabilities, average="macro")
    weighted_auc = roc_auc_score(labels_bin, all_probabilities, average="weighted")
    micro_auc = roc_auc_score(labels_bin, all_probabilities, average="micro")

    print(f"\nMacro-average AUC:    {macro_auc:.4f}")
    print(f"Weighted-average AUC: {weighted_auc:.4f}")
    print(f"Micro-average AUC:    {micro_auc:.4f}")


def main():

    parser = argparse.ArgumentParser(
        description="Run inference with the trained reference-configuration model, without retraining."
    )
    parser.add_argument(
        "--patient_dir", type=str, default=None,
        help="Path to one preprocessed patient folder (T2.nii.gz, Pre.nii.gz, Post1.nii.gz)"
    )
    parser.add_argument(
        "--test_set", action="store_true",
        help="Evaluate on the processed test split (reproduces the reported reference-configuration test metrics)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=CHECKPOINT_PATH,
        help=f"Path to a model checkpoint (default: {CHECKPOINT_PATH})"
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    if args.patient_dir:

        predicted_class, probabilities = predict_patient(model, args.patient_dir)

        print(f"Predicted class: {CLASS_NAMES[predicted_class]}\n")

        for name, prob in zip(CLASS_NAMES, probabilities):
            print(f"  {name:<12} {prob:.4f}")

    elif args.test_set:

        evaluate_test_set(model)

    else:

        parser.print_help()


if __name__ == "__main__":
    main()
