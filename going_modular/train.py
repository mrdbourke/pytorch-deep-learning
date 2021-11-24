import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import data_setup, engine, model_builder

if __name__ == "__main__":
    ## Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Setup data
    train_dir = "data/10_whole_foods/train"
    test_dir = "data/10_whole_foods/test"

    # Create transforms
    data_transform = transforms.Compose(
        [transforms.Resize([64, 64]), transforms.ToTensor()]
    )

    # Create data loaders
    print(f"Loading data from: {train_dir} and {test_dir}...")
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=32
    )

    ## Setup model
    print(f"Building model...")
    model = model_builder.TinyVGG(
        input_size=3, hidden_units=10, output_shape=len(class_names)
    ).to(device)

    ## Train model

    # Set loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    print(f"Starting training...")
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        epochs=5,
    )

    ## Save model

    # Save to file
    os.makedirs("models", exist_ok=True)
    save_path = "models/04_pytorch_custom_datasets_tinyvgg.pth"
    print(f"Finished training, saving model to {save_path}...")
    torch.save(
        model.state_dict(), save_path
    )  # only saving the state_dict() only saves the model's learned parameters
