"""
Parallel model definition for the two-stage (hierarchical) cascade
classification experiment: instead of one 3-way classifier, cascade two
binary classifiers --

    Stage 1: No lesion  vs.  Lesion        (Benign + Malignant collapsed)
    Stage 2: Benign     vs.  Malignant     (lesion-positive patients only)

Architecture is otherwise byte-for-byte identical to model.py's
BreastMRINetwork (ResNet18, ImageNet-pretrained, layer4-only unfrozen,
plain attention pooling, same classifier head shape) -- only the final
classifier layer's output width changes (2 instead of 3), since each
stage is a binary problem. One instance of this network is trained per
stage (see train_stage1.py / train_stage2.py). model.py is untouched.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # remove final FC layer
        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False

        # unfreeze only the last residual block (layer4) -- matches the reference configuration
        for param in self.features[-2].parameters():
            param.requires_grad = True

    def train(self, mode=True):

        super().train(mode)

        # keep the still-frozen layers (everything except layer4) in
        # eval mode, so their BatchNorm running stats don't drift
        if mode:
            unfrozen = {len(self.features) - 2}
            for i, layer in enumerate(self.features):
                if i not in unfrozen:
                    layer.eval()

        return self

    def forward(self, x):

        # x: [batch*slices, 3, 128, 128]
        x = self.features(x)

        # [batch*slices, 512, 1, 1]
        x = x.flatten(1)

        # [batch*slices, 512]
        return x


class AttentionPooling(nn.Module):

    """Attention-weighted pooling: scores each slice's feature vector with a
    small MLP, then combines slices into one patient-level feature via a
    softmax-weighted sum."""

    def __init__(self, feature_dim=512, hidden_dim=128):

        super().__init__()

        self.attention_scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):

        """
        features: (batch_size, num_slices, feature_dim)
        returns:  (batch_size, feature_dim)
        """

        attention_scores = self.attention_scorer(features)

        # (batch_size, num_slices, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        pooled_feature = torch.sum(attention_weights * features, dim=1)

        return pooled_feature


class Classifier(nn.Module):

    """Same head shape as model.py's Classifier, but num_classes is
    parameterized (2 for either stage of the cascade, instead of the
    reference configuration's hardcoded 3)."""

    def __init__(self, num_classes=2):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        return self.classifier(x)


class BreastMRINetworkHierarchical(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.feature_extractor = ResNet18FeatureExtractor()

        self.attention_pool = AttentionPooling()

        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, patient_volume):
        """
        patient_volume shape: (batch_size, num_slices, 3, 128, 128)
        """

        batch_size = patient_volume.shape[0]
        num_slices = patient_volume.shape[1]

        # Merge batch and slice dimensions
        x = patient_volume.reshape(batch_size * num_slices, 3, 128, 128)

        # Feature extraction: (batch*slices, 512)
        x = self.feature_extractor(x)

        # Restore patient dimension: (batch, 32, 512)
        x = x.reshape(batch_size, num_slices, 512)

        # Attention pooling: (batch, 512)
        x = self.attention_pool(x)

        # Classification: (batch, num_classes)
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":

    # sanity check, mirrors model.py's self-test

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BreastMRINetworkHierarchical(num_classes=2).to(device)

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