import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

DATA_ROOT = "data"


def load_images(name, split="train"):
    """Return uint8 array (N, 28, 28) and labels (N,)."""
    if name == "mnist":
        ds = datasets.MNIST(DATA_ROOT, train=(split == "train"), download=True)
        return ds.data.numpy(), ds.targets.numpy()
    if name == "pneumonia":
        import medmnist
        ds = medmnist.PneumoniaMNIST(split=split, download=True, root=DATA_ROOT, size=28)
        return ds.imgs, ds.labels.reshape(-1)
    raise ValueError(name)


def to_tensor(imgs):
    """uint8 [0,255] -> float [-1,1], shape (N,1,28,28)."""
    return torch.from_numpy(imgs.astype(np.float32) / 127.5 - 1.0).unsqueeze(1)


def make_dataset(name, split="train"):
    imgs, labels = load_images(name, split)
    return TensorDataset(to_tensor(imgs), torch.from_numpy(labels).long())
