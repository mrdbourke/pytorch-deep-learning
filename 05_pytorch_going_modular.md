# TODO: Going modular

In this section we're going to turn notebook 04 into a series of Python scripts saved to a directory called `going_modular`.

This document will describe those steps.

Afterwards we can reuse the same code in the scripts in the next notebook (06. Transfer Learning).

## TK - What is going modular?

Going modular involves turning notebook code (from a Jupyter Notebook or Google Colab notebook) into a series of different Python scripts that offer similar functionality.

For example, we'd turn our notebook code from a series of cells into the following files:
* `data_setup.py` - a file to prepare and download data if needed.
* `engine.py` - a file containing various training functions.
* `model_builder.py` or `model.py` - a file to create a PyTorch model.
* `train.py` - a file to leverage all other files and train a target PyTorch model.
* `utils.py` - a file dedicated to helpful utility functions.

> **Note:** The naming and layout of the above files will depend on your use case and code requirements. Python scripts are as general as individual notebook cells, meaning, you could create one for almost any kind of functionality.

## TK - Why would you want to go modular?

Notebooks are fantastic for iteratively exploring and running experiments quickly.

However, for larger scale projects you may find Python scripting more reproducible and easier to run.

Though this is a debated topic, as companies like [Netflix have shown they can use notebooks for production code](https://netflixtechblog.com/notebook-innovation-591ee3221233).

**Production code** is code that runs to offer a service to someone or something.

For example, if you have an app running online that other people can access and use, the code running that app is considered **production code**.

And libraries like fast.ai's [`nb-dev`](https://github.com/fastai/nbdev) (short for notebook development) enable you to write whole Python libraries (including documentation) with Jupyter Notebooks.

### Pros and cons of notebooks vs Python scripts

There's arguments for both sides.

But this list sums up a few of the main topics.

| | **Pros** | **Cons** | 
| ----- | ----- | ----- |
| **Notebooks** | Easy to experiment/get started | Versioning can be hard |
| | Easy to share (e.g. a link to a Google Colab notebook) | Hard to use only specific parts |
| | Very visual | Text and graphics can get in the way of code | 
| | | |
| **Python scripts** | Can package code together (saves rewriting similar code across different notebooks) | Experimenting isn't as visual/have to run the whole script (rather than one cell) |
| | Can use git for versioning | |
| | Many open source projects use scripts | |  
| | Larger projects can be run on cloud vendors (not as much support for notebooks) | | 

### My workflow

I start ML projects in Jupyter/Google Colab notebooks for quick experimentation and visualization.

Then when I've got something working, I move to Python scripts.

### PyTorch in the wild

In your travels, you'll see many code repositories for PyTorch-based ML projects have instructions on how to run the PyTorch code in the form of Python scripts.

For example, you might be instructred to run code like the following in a terminal/command line to train a model:

```
train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

TK image - annotate the above line 

In this case, `train.py` is the target Python script, it'll likely contain functions to train a PyTorch model.

And `--model`, `--batch_size`, `--lr` and `--num_epochs` are known as argument flags.

You can set these to whatever values you like and if they're compatible with `train.py`, they'll work, if not, they'll error.

For example, let's say we wanted to train our TinyVGG model for 10 epochs with a batch size of 32 and a learning rate of 0.001:

```
train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 32
```

You could setup any number of these argument flags in your `train.py` script to suit your needs.

The PyTorch blog post for training state-of-the-art computer vision models uses this style.

TK image for PyTorch blog post training code - https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#break-down-of-key-accuracy-improvements

## TK - What we'ere going to cover

TODO
* TK - part 1, part 2 notebooks
* What are we going to end up with? What is the end file structure?

* Possible workflow: start in notebooks, explore and get something working, then go to Python scripts
* Can use "%%writefile path/to/python/script.py" as jupyter cell magic to write a Python code file 
* What should be the environment used for teaching the refactoring section? Could everyone use VSCode or Jupyter Lab? Or VS Code?
* TK - showcase how quickly notebook 04 can be reproduced when everything in a modular fashion (call the scripts on the command line).. e.g. `train_model.py...` -> saved model in target folder

## TK - Where can you get help?

All of the materials for this course [are available on GitHub](https://github.com/mrdbourke/pytorch-deep-learning).

If you run into trouble, you can ask a question on the course [GitHub Discussions page](https://github.com/mrdbourke/pytorch-deep-learning/discussions).

And of course, there's the [PyTorch documentation](https://pytorch.org/docs/stable/index.html) and [PyTorch developer forums](https://discuss.pytorch.org/), a very helpful place for all things PyTorch. 

## TODO:
* Go through each of the different sections and explain what's happening
* How to go from functions in a notebook to `.py` (manaul vs automatic)
    * e.g. why use %%writefile vs just copy and paste?

## TK - 0. Cell mode vs. script mode

## TK - 1. Get data

## TK - 2. Create Datasets and DataLoaders

## TK - 3. Setting up the training code (the engine)

## TK - 4. Creating `train_step()` and `test_step()` functions and `train()` to combine them  

## TK -  5. Creating a function to save the model

## TK - 6. Train, evaluate and save the model 

## TK - Exercises
* TK Add an argument for using a different:
    * train/test dir
    * optimizer
    * etc...
* Use `argparse` to be able to send `train.py` custom settings for training procedures

## TK - Extra-curriculum
* Read up on structuring a Python project on RealPython - https://realpython.com/python-application-layouts/ 
* Recommended code structure for training your PyTorch model - https://github.com/IgorSusmelj/pytorch-styleguide#recommended-code-structure-for-training-your-model 