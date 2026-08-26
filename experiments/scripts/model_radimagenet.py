"""
Fully-frozen RadImageNet-pretrained ResNet50 backbone (the frozen-RadImageNet-baseline
diagnostic setup, kept frozen for the augmented frozen-RadImageNet run too).
That augmented run changes only the trainable head's training recipe to
match the reference configuration's (lr=1e-5, weight_decay=1e-4, dropout
0.5) and adds augmentation - backbone freezing is unchanged, so any
difference vs. the frozen-RadImageNet baseline isolates the effect of those
head-side changes alone.

Weights: Lab-Rasool/RadImageNet (Hugging Face mirror, MIT license),
ResNet50.pt is a state_dict for a standard torchvision resnet50, keys
prefixed "backbone." and covering conv1/bn1/layer1-4 only (no avgpool/fc
params to load, since those layers have none). Confirmed by direct
inspection: strict-loads into nn.Sequential(*resnet50().children()[:-1]).

Preprocessing: RadImageNet's original training pipeline normalizes with
(pixel - 127.5) * 2 / 255, i.e. raw [0,255] pixels -> [-1,1].
Since dataset.py's ODELIADataset already min-max normalizes each volume to
[0,1], the equivalent remap here is a simple x -> 2x - 1, applied in
forward() so dataset.py doesn't need to change.
"""

import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download

RADIMAGENET_REPO_ID = "Lab-Rasool/RadImageNet"
RADIMAGENET_FILENAME = "ResNet50.pt"


class RadImageNetResNet50FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights=None)

        # remove final FC layer, keep avgpool
        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        weights_path = hf_hub_download(
            repo_id=RADIMAGENET_REPO_ID,
            filename=RADIMAGENET_FILENAME
        )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = {k.replace("backbone.", "", 1): v for k, v in state_dict.items()}

        self.features.load_state_dict(state_dict, strict=True)

        # fully frozen backbone (unchanged from the frozen-RadImageNet
        # baseline - only the head's training recipe changes for the
        # augmented frozen-RadImageNet run)
        for param in self.features.parameters():
            param.requires_grad = False

    def train(self, mode=True):

        super().train(mode)

        # backbone is fully frozen, so it always stays in eval mode
        self.features.eval()

        return self

    def forward(self, x):

        # x:
        # [batch*slices, 3, 128, 128], values in [0, 1] (ODELIADataset convention)

        x = x * 2.0 - 1.0

        # RadImageNet-expected range: [-1, 1]

        x = self.features(x)

        # [batch*slices, 2048, 1, 1]

        x = x.flatten(1)

        # [batch*slices, 2048]

        return x


class AttentionPooling(nn.Module):

    """Attention-weighted pooling: scores each slice's feature vector with a
    small MLP, then combines slices into one patient-level feature via a
    softmax-weighted sum."""

    def __init__(self, feature_dim=2048):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, features):

        """
        features

        shape:
        (batch_size, num_slices, feature_dim)

        returns

        (batch_size, feature_dim)
        """

        attention_scores = self.attention(features)

        # (batch_size, num_slices, 1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        pooled_feature = torch.sum(
            attention_weights * features,
            dim=1
        )

        return pooled_feature


class Classifier(nn.Module):

    def __init__(self, feature_dim=2048):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):

        return self.classifier(x)


class BreastMRINetworkRadImageNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.feature_extractor = RadImageNetResNet50FeatureExtractor()

        self.attention_pool = AttentionPooling(feature_dim=2048)

        self.classifier = Classifier(feature_dim=2048)

    def forward(self, patient_volume):
        """
        patient_volume

        shape:

        (batch_size,
         num_slices,
         3,
         128,
         128)
        """

        batch_size = patient_volume.shape[0]

        num_slices = patient_volume.shape[1]

        # -----------------------------------------
        # Merge batch and slice dimensions
        # -----------------------------------------

        x = patient_volume.reshape(

            batch_size * num_slices,

            3,

            128,

            128

        )

        # shape:
        # (batch*slices,3,128,128)

        # -----------------------------------------
        # Feature extraction
        # -----------------------------------------

        x = self.feature_extractor(x)

        # shape:
        # (batch*slices,2048)

        # -----------------------------------------
        # Restore patient dimension
        # -----------------------------------------

        x = x.reshape(

            batch_size,

            num_slices,

            2048

        )

        # shape:
        # (batch,32,2048)

        # -----------------------------------------
        # Attention pooling
        # -----------------------------------------

        x = self.attention_pool(x)

        # shape:
        # (batch,2048)

        # -----------------------------------------
        # Classification
        # -----------------------------------------

        logits = self.classifier(x)

        # shape:
        # (batch,3)

        return logits


if __name__ == "__main__":

    # sanity check, mirrors model.py's self-test

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BreastMRINetworkRadImageNet().to(device)

    fake_batch = torch.randn(16, 32, 3, 128, 128).to(device)

    output = model(fake_batch)

    print(output.shape)

    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        total_params += p.numel()

        if p.requires_grad:
            trainable_params += p.numel()

    print("Total:", total_params)
    print("Trainable:", trainable_params)