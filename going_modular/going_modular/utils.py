"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path

import torch

import model_builder

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def load_model(model_dir: str,
               model_name: str,
               hidden_units: int):
    """Loads a PyTorch model from a target directory.

    Args:
    model_dir: A directory where the model is located.
    model_name: The name of the model to load.
      Should include either ".pth" or ".pt" as the file extension.
    hidden_uints: number of hidden units in the model.

    Example usage:
    model = load_model(model_dir="models",
                       model_name="05_going_modular_tingvgg_model.pth",
                       hidden_units=128)

    Returns:
    The loaded PyTorch model.
    """
    # Create the model directory path
    model_dir_path = Path(model_dir)

    # Create the model path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = model_dir_path / model_name

    # Load the model
    print(f"[INFO] Loading model from: {model_path}")
    
    # Build the model with pre-trained parameters
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=hidden_units,
        output_shape=3
        )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model
