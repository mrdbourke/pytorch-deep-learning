
"""
Predicts the class of an input image using a trained TinyVGG model. Posible class names are: pizza, steak, sushi.
"""

import os
import torch
import argparse
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import model_builder, utils

def get_args_parser(add_help=True):

    # Argument parser for input arguments
    parser = argparse.ArgumentParser(description="Predicts the class of an input image using a trained TinyVGG model. Posible class names are: pizza, steak, sushi.", add_help=add_help)

    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--image_name', type=str, required=True, help="Image name.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name.")
    parser.add_argument('--hidden_units', type=int, default=None, help="Number of hidden units per stage.")

    return parser

def main(args):
    """
    Predicts the class of an input image using a trained TinyVGG model.
    """

    # Create class_names (one output unit unit)
    class_names = ["pizza", "steak", "sushi"]

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = os.path.join(args.image_path, args.image_name)

    print(f"[INFO] Predicting on {image_path}")

    target_image = torchvision.io.read_image(str(image_path))

    # Create the transform for the input image
    data_transform = v2.Compose([
        v2.Resize((64, 64)),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # And execute the transform!
    target_image = data_transform(target_image)

    model_path = os.path.join(args.model_path, args.model_name)
    
    # Build the model with pre-trained parameters
    model = utils.load_model(model_dir=args.model_path, model_name=args.model_name, hidden_units=args.hidden_units)

    model.to(device)
    
    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Obtain the class name
    pred_image_label_class = class_names[target_image_pred_label]
    
    print(f"[INFO] Pred class: {pred_image_label_class}, Pred prob: {target_image_pred_probs.max():.3f}")


if __name__ == '__main__':    
    args = get_args_parser().parse_args()
    main(args)  
