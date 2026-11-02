import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import INFO


def load_datasets():
    """Load PneumoniaMNIST using the preprocessing in the uploaded notebook."""
    from medmnist import PneumoniaMNIST
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    train = PneumoniaMNIST(split="train", transform=transform, download=True)
    val = PneumoniaMNIST(split="val", transform=transform, download=True)
    test = PneumoniaMNIST(split="test", transform=transform, download=True)
    return train, val, test


def make_loaders(train, val, test, batch_size=8):
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def iid_client_subsets(train_dataset, num_clients=3, seed=42, use_torch_permutation=False):
    """Create reproducible IID partitions; each training example belongs to one client."""
    if use_torch_permutation:
        indices = torch.randperm(
            len(train_dataset),
            generator=torch.Generator().manual_seed(seed)
        ).tolist()
    else:
        indices = np.random.default_rng(seed).permutation(len(train_dataset)).tolist()
    parts = np.array_split(indices, num_clients)
    return [Subset(train_dataset, part.tolist()) for part in parts]
