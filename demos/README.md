# FoodVision Demos
The `.zip` files contained in this folder contain the files for creating the FoodVision Mini and FoodVision Big demos.

The code to create these can be found in [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/).

Each `.zip` folder contains: 
* `model.py` - a file defining the model architecture used for the demo.
* `app.py` - a file defining the Gradio demo application.
* `*.pth` - a trained PyTorch model file with the same architecture as in `model.py`.
* `requirements.txt` - a list of required packages for deploying the app.
* `examples/` - a folder containing various image examples to try with the demo.

You can see a deployed version of each demo on HuggingFace Spaces:

| **App** | **What does it do?** | **Live demo** | **See files** | **Download files** |
| ----- | ----- | ----- | ----- | ----- |
| FoodVision Mini | Classifies an image of food into pizza 🍕, steak 🥩 or sushi 🍣.  | [Demo](https://huggingface.co/spaces/mrdbourke/foodvision_mini) | [Files](https://huggingface.co/spaces/mrdbourke/foodvision_mini/tree/main) | [Download `foodvision_mini.zip`](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/demos/foodvision_mini.zip) |
| FoodVision Big 💪 | Classifies an image of food into 101 different food classes from the Food101 dataset. | [Demo](https://huggingface.co/spaces/mrdbourke/foodvision_big/) | [Files](https://huggingface.co/spaces/mrdbourke/foodvision_big/tree/main) | [Download `foodvision_big.zip`](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/demos/foodvision_big.zip) |

