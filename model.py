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

        # unfreeze only the last residual block (layer4)
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

        # x:
        # [batch*slices, 3, 128, 128]

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

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
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

    def __init__(self):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):

        return self.classifier(x)


class BreastMRINetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.feature_extractor = ResNet18FeatureExtractor()

        self.attention_pool = AttentionPooling()

        self.classifier = Classifier()

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
        # (batch*slices,512)

        # -----------------------------------------
        # Restore patient dimension
        # -----------------------------------------

        x = x.reshape(

            batch_size,

            num_slices,

            512

        )

        # shape:
        # (batch,32,512)

        # -----------------------------------------
        # Attention pooling
        # -----------------------------------------

        x = self.attention_pool(x)

        # shape:
        # (batch,512)

        # -----------------------------------------
        # Classification
        # -----------------------------------------

        logits = self.classifier(x)

        # shape:
        # (batch,3)

        return logits


if __name__ == "__main__":

    # smoke test: random batch through the network, and a check that
    # only layer4 (+ attention/classifier heads) is trainable
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BreastMRINetwork().to(device)

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