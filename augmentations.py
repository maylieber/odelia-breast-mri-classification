"""
Each transform operates on a single patient's volume, shaped
(32, 3, H, W) — 32 slices, 3 modalities (T2 / Pre / Post1) per slice.
Flip/noise/contrast are applied with independent probabilities (each
one may or may not trigger per sample), not as mutually exclusive
choices.
"""

import random
import torch


class RandomHorizontalFlip:

    """
    Flips the width axis with probability p. One flip decision is made
    for the whole volume (all slices, all modalities) since they're
    the same anatomy and must stay spatially consistent.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):

        if random.random() < self.p:
            images = torch.flip(images, dims=[-1])

        return images


class RandomGaussianNoise:

    """Adds zero-mean Gaussian noise with probability p."""

    def __init__(self, p=0.5, std=0.05):
        self.p = p
        self.std = std

    def __call__(self, images):

        if random.random() < self.p:
            noise = torch.randn_like(images) * self.std
            images = torch.clamp(images + noise, 0.0, 1.0)

        return images


class RandomContrast:

    """
    Scales intensities around the volume's own mean by a random
    factor, with probability p.
    """

    def __init__(self, p=0.5, factor_range=(0.8, 1.2)):
        self.p = p
        self.factor_range = factor_range

    def __call__(self, images):

        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            mean = images.mean()
            images = torch.clamp((images - mean) * factor + mean, 0.0, 1.0)

        return images


class Compose:

    """Applies a list of transforms in sequence."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):

        for transform in self.transforms:
            images = transform(images)

        return images


def train_augmentations():

    """
    Standard augmentation pipeline for the training split.
    Not meant to be applied to validation/test data.
    """

    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomGaussianNoise(p=0.3, std=0.05),
        RandomContrast(p=0.3, factor_range=(0.8, 1.2)),
    ])