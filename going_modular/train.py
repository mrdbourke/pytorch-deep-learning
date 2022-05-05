import os
import torch
import argparse
import data_setup, engine, model_builder

from torch import nn
from torchvision import transforms

parser = argparse.ArgumentParser(
    description="Food Vision Mini Classification Training", add_help=True
)

parser.add_argument("--model", default="tinyvgg", type=str, help="model to use")
parser.add_argument(
    "-b", "--batch_size", default=32, type=int, help="images to use per batch"
)
parser.add_argument("--lr", default=0.001, type=float, help="learning rate to use")
parser.add_argument(
    "-n", "--num_epochs", default=10, type=int, help="number of epochs to train for"
)

# Get the arguments from the parser
args = parser.parse_args()

# Run the training code if the script is the main script (e.g. python train.py...)
if __name__ == "__main__":
    ## Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} for computing...")

    ## TK - turn into argparse for train/test dir Setup data
    train_dir = "../data/pizza_steak_sushi/train"
    test_dir = "../data/pizza_steak_sushi/test"

    # Create transforms
    data_transform = transforms.Compose(
        [transforms.Resize([64, 64]), transforms.ToTensor()]
    )

    # Create data loaders
    print(
        f"[INFO] Loading train data from: {train_dir}\n"
        f"[INFO] Loading test data from: {test_dir}\n"
        f"[INFO] Using batch size: {args.batch_size}"
    )
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=args.batch_size,
    )

    ## Setup model
    if args.model == "tinyvgg":
        print(f"[INFO] Building {args.model} model")
        model = model_builder.TinyVGG(
            input_size=3, hidden_units=24, output_shape=len(class_names)
        ).to(device)
    else:
        print("'{args.model}' not found, please pass in a valid model name, exiting...")
        exit()

    ## Train model

    # Set loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # Start training
    print(f"[INFO] Starting training for {args.num_epochs} epochs")
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.num_epochs,
        device=device,
    )

    ## Save model

    # TK - change to argparse for where to save model Save to file
    os.makedirs("models", exist_ok=True)
    save_path = "models/04_pytorch_custom_datasets_tinyvgg.pth"
    print(f"Finished training, saving model to {save_path}...")
    torch.save(
        model.state_dict(), save_path
    )  # only saving the state_dict() only saves the model's learned parameters
