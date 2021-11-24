import os

from torchvision import datasets
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir, test_dir, transform, batch_size=32, num_workers=NUM_WORKERS
):
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    return train_dataloader, test_dataloader, class_names
