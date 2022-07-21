# Going Modular Scripts

The Python scripts in this directory were generated using the notebook [05. Going Modular Part 2 (script mode)](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb).

They breakdown as follows: 
* `data_setup.py` - a file to prepare and download data if needed.
* `engine.py` - a file containing various training functions.
* `model_builder.py` - a file to create a PyTorch TinyVGG model.
* `train.py` - a file to leverage all other files and train a target PyTorch model.
* `utils.py` - a file dedicated to helpful utility functions.
* **Extra:** `predictions.py` - a file for making predictions with a trained PyTorch model and input image (the main function, `pred_and_plot_image()` was originally created in [06. PyTorch Transfer Learning section 6](https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set)).

For an explanation of how this was done, refer to section [05. PyTorch Going Modular of the learnpytorch.io book](https://www.learnpytorch.io/05_pytorch_going_modular/).