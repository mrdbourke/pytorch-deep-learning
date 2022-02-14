# Getting setup to code PyTorch

Setting a machine for coding deep learning can be quite involved.

From hardware to software to all of the little bits and pieces to make your code run on someone else's machine just like it ran on yours.

For the sake of this course, we're keeping things simple.

Though not so simple you won't be able to use what you're using here elsewhere.

There's two setup options. One is easier than the other but the latter offers more options in the long-run.

1. Use Google Colab (easiest)
2. Setup on your own local/remote machine (a few steps but you've got a bit more flexibility here)

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

### Get started with Google Colab

For the starter notebooks of the course (00-04), we'll be using exclusively Google Colab.

This is because it more than satisfies our needs.

In fact, this is the workflow I'll often do myself.

I do a large amount of beginner and experimental work in Google Colab.

And when I've found something I'd like to turn into a larger project or work on more, I move to local compute or cloud-hosted compute.

To begin with Google Colab, I'd first go through the [Introduction to Google Colab notebook](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) (just to get familiar with all the bells and buttons).

After that, you can run any of the course notebooks directly in Google Colab by pressing the "Open in Colab" button at the top of the online book version or the GitHub version.

![open a course notebook in Google Colab via open in Colab button](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-in-colab-cropped.gif)

If you'd like to make a copy of the notebook and store it on your Google Drive, you can press the "Copy to Drive" button.

You can also enter a notebook link from GitHub directly in Google Colab and get the same result.

![open a course notebook in Google Colab via GitHub link](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/setup-open-notebook-in-colab-via-link.png)

Doing this will give you a runable notebook right in Google Colab. 

Though this should only be used for testing purposes, as when going through the course, I highly recommend you **write the code yourself**, rather than running existing code.

### TK - Getting acess to a GPU in Google Colab

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

TK - this is not a replacement for the PyTorch documentation for getting setup locally: https://pytorch.org/get-started/locally/