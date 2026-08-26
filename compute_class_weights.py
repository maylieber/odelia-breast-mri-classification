import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def compute_weights(PROCESSED_MASTER_TABLE):

    df = pd.read_csv(PROCESSED_MASTER_TABLE)

    train_df = df[df["Split"] == "train"]

    labels = train_df["Lesion"].values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=labels
    )

    weights = torch.tensor(weights, dtype=torch.float32)

    return weights


if __name__ == "__main__":

    weights = compute_weights()

    print("Class weights:")
    print(weights)
