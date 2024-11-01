"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
from torchvision import transforms
import data_setup, engine, model_builder, utils

def get_args_parser(add_help=True):
  
  # Argument parser for input arguments
  parser = argparse.ArgumentParser(description="Train a TinyVGG model on a custom dataset.", add_help=add_help)

  # Define arguments
  parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for.")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
  parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in the model.")
  parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
  parser.add_argument("--train_dir", type=str, default="data/pizza_steak_sushi/train", help="Directory for training data.")
  parser.add_argument("--test_dir", type=str, default="data/pizza_steak_sushi/test", help="Directory for testing data.")
  parser.add_argument("--model_dir", type=str, default="models", help="Directory to save the trained model.")
  parser.add_argument("--model_name", type=str, default="05_going_modular_script_mode_tinyvgg_model.pth", help="Name for the saved model.")

  return parser

def main(args):
  
  # Set up device
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  
  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=args.train_dir,
      test_dir=args.test_dir,
      transform=data_transform,
      batch_size=args.batch_size
  )

  #print(f"Number of training samples: {len(train_dataloader)}")
  #print(f"Number of testing samples: {len(test_dataloader)}")
  #print(f"Class names: {class_names}")
  #print(f"Device: {device}")

  # Create model with help from model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=args.hidden_units,
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=args.learning_rate)

  # Start training with help from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=args.num_epochs,
              device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                  target_dir=args.model_dir,
                  model_name=args.model_name) #"05_going_modular_script_mode_tinyvgg_model.pth")

if __name__ == '__main__':
  args = get_args_parser().parse_args()
  main(args)  
