# Getting setup to code PyTorch

Setting up a machine for coding deep learning can be quite involved.

From hardware to software to all of the little bits and pieces to make your code run on someone else's machine just like it ran on yours.

For the sake of this course, we're keeping things simple.

Though not so simple you won't be able to use what you're using here elsewhere.

There are two setup options. One is easier than the other but the other offers more options in the long-run.

1. Use Google Colab (easiest)
2. Setup on your own local/remote machine (a few steps but you've got a bit more flexibility here)

**Note** Neither of these are replacements for [PyTorch's official setup docs](https://pytorch.org/get-started/locally/), if you're wanting to start coding PyTorch for the long term, you should get familiar with those.

## 1. Setup with Google Colab (easiest)

Google Colab is a free online interactive compute engine (based on Jupyter Notebooks, the data science standard).

The benefits of Google Colab are:
* Almost zero setup (Google Colab comes with PyTorch and many other data science packages such as pandas, NumPy and Matplotlib already installed)
* Share your work with a link
* Free access to GPUs (GPUs make your deep learning code faster), with a paid option to access *more* GPU power

The cons of Google Colab are:
* Timeouts (most Colab notebooks only preserve state for 2-3 hours max, though this can increase with the paid option)
* No access to local storage (though there are ways around this)
* Not as well setup for scripting (turning your code into modules)

### Use Google Colab to begin, scale up when needed

For the starter notebooks of the course (00-04), we'll be using exclusively Google Colab.

This is because it more than satisfies our needs.

In fact, this is the workflow I'll often do myself.

I do a large amount of beginner and experimental work in Google Colab.

And when I've found something I'd like to turn into a larger project or work on more, I move to local compute or cloud-hosted compute.

### Getting started with Google Colab

To begin with Google Colab, I'd first go through the [Introduction to Google Colab notebook](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) (just to get familiar with all the bells and buttons).

### Opening a course notebook with one-click

After you've gotten familiar with the Google Colab interface, you can run any of the course notebooks directly in Google Colab by pressing the "Open in Colab" button at the top of the online book version or the GitHub version.

![open a course notebook in Google Colab via open in Colab button](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-in-colab-cropped.gif)

If you'd like to make a copy of the notebook and store it on your Google Drive, you can press the "Copy to Drive" button.

### Opening a notebook in Google Colab with a link

You can also enter any notebook link from GitHub directly in Google Colab and get the same result.

![open a course notebook in Google Colab via GitHub link](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-notebook-in-colab-via-link.png)

Doing this will give you a runable notebook right in Google Colab. 

Though this should only be used for testing purposes, as when going through the course, I highly recommend you **write the code yourself**, rather than running existing code.

### Getting acess to a GPU in Google Colab

To get access to a CUDA-enabled NVIDIA GPU (CUDA is the programming interface that allows deep learning code to run faster on GPUs) in Google Colab you can go to `Runtime -> Change runtime type -> Hardware Accelerator -> GPU` (note: this will require the runtime to restart).

![Getting access to a GPU in Google Colab](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-get-gpu-colab-cropped.gif)

To check whether you have a GPU active in Google Colab you can run:

```
!nvidia-smi
```

If you have access to a GPU, this will show you what kind of GPU you have access to.

And to see if PyTorch has access to the GPU, you can run:

```python
import torch # Google Colab comes with torch already installed
print(torch.cuda.is_available()) # will return True if PyTorch can use the GPU
```

If PyTorch can see the GPU on Google Colab, the above will print `True`.

## TK - 2. Getting setup locally (Linux version)

> **Note:** A reminder this is not a replacement for the [PyTorch documentation for getting setup locally](https://pytorch.org/get-started/locally/). This is only one way of getting setup (there are many) and designed specifically for this course.

This **setup is focused on Linux systems** (the most common operating system in the world), if you are running Windows or macOS, you should refer to the PyTorch documentation. 

This setup also **expects you to have access to a NVIDIA GPU**.

Why this setup?

As a machine learning engineer, I use it almost daily. It works for a large amount of workflows and it's flexible enough so you can change if you need.

Let's begin.

### Setup steps locally for a Linux system with a GPU
TK TODO - add step for install CUDA drivers
TK image - overall setup of the course environment (e.g. Jupyter Lab inside conda env)

1. [Install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (you can use Anaconda if you already have it), the main thing is you need access to `conda` on the command line. Make sure to follow all the steps in the Miniconda installation guide before moving onto the next step.
2. Make a directory for the course materials, you can name it what you want and then change into it. For example:
```
mkdir ztm-pytorch-course
cd ztm-pytorch-course
```
3. Create a `conda` environment in the directory you just created. The following command will create a `conda` enviroment that lives in the folder called `env` which lives in the folder you just created (e.g. `ztm-pytorch-course/env`). Press `y` when the command below asks `y/n?`.
```
conda create --prefix ./env python=3.8.13
```
4. Activate the environment you just created.
```
conda activate ./env
```
5. Install the code dependencies you'll need for the course such as PyTorch and CUDA Toolkit for running PyTorch on your GPU. You can run all of these at the same time (**note:** this is specifically for Linux systems with a NVIDIA GPU, for other options see the [PyTorch setup documentation](https://pytorch.org/get-started/locally/)):
```
conda install -c pytorch pytorch=1.10.0 torchvision cudatoolkit=11.3 -y
conda install -c conda-forge jupyterlab torchinfo torchmetrics -y
conda install -c anaconda pip -y
conda install pandas matplotlib scikit-learn -y
```
6. Verify the installation ran correctly by running starting a Jupyter Lab server:

```bash
jupyter lab
```

7. After Jupyter Lab is running, start a Jupyter Notebook and running the following piece of code in a cell.
```python
import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo, torchmetrics

# Check PyTorch access (should print out a tensor)
print(torch.randn(3, 3))

# Check for GPU (should return True)
print(torch.cuda.is_available())
```

If the above code runs without errors, you should be ready to go.

If you do run into an error, please refer to the [Learn PyTorch GitHub Discussions page](https://github.com/mrdbourke/pytorch-deep-learning/discussions) and ask a question or the [PyTorch setup documentation page](https://pytorch.org/get-started/locally/).