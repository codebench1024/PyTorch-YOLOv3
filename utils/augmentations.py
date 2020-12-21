import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision.transforms as transforms

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def vertical_flip(images, targets):
    images = torch.flip(images, [-2])
    targets[:, 3] = 1 - targets[:, 3]
    return images, targets

def rotation_90(images, targets):
    images = torch.rot90(images, 1, dims=(1, 2))
    tmp = torch.clone(targets)
    targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5] = tmp[:, 3], 1 - tmp[:, 2], tmp[:, 5], tmp[:, 4]
    return images, targets

def rotation_180(images, targets):
    images = torch.rot90(images, 2, dims=(1, 2))
    targets[:, 2] = 1 - targets[:, 2]
    targets[:, 3] = 1 - targets[:, 3]
    return images, targets

def rotation_270(images, targets):
    images = torch.rot90(images, 3, dims=(1, 2))
    tmp = torch.clone(targets)
    targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5] = 1 - tmp[:, 3], tmp[:, 2], tmp[:, 5], tmp[:, 4]
    return images, targets

def random_bright(images, targets):
    # delta = 0.1
    # delta = random.uniform(-delta, delta)
    # images = images + delta
    # images
    trans = transforms.ColorJitter(brightness=0.1, contrast=0.1)
    images = trans(images)
    return images, targets
