# TODO: Going modular

In this section we're going to turn notebook 04 into a series of Python scripts saved to a directory called `going_modular`.

This document will describe those steps.

Afterwards we can reuse the same code in the scripts in the next notebook (06), Transfer Learning.

## TK - What is going modular?

Going modular involves turning notebook code (from a Jupyter Notebook or Google Colab notebook) into a series of different Python scripts that offer similar functionality.

For example, we'd turn our notebook code from a series of cells into the following files:
* `data_setup.py` - a file to prepare and download data if needed.
* `model_builder.py` or `model.py` - a file to create a PyTorch model.
* `engine.py` - a file containing various training functions.
* `train.py` - a file to leverage all other files and train a target PyTorch model.
* `utils.py` - a file dedicated to helpful utility functions.

> **Note:** The naming and layout of the above files will depend on your use case and code requirements. Python scripts are as general as individual notebook cells, meaning, you could create one for almost any kind of functionality.

## TK - Why would you want to go modular?

Notebooks are fantastic for iteratively exploring and running experiments quickly.

However, for larger scale projects you may find Python scripting more reproducible and easier to run.

Though this is a debated topic, as companies like [Netflix have shown they can use notebooks for production code](https://netflixtechblog.com/notebook-innovation-591ee3221233).

**Production code** is code that runs to offer a service to someone or something.

For example, if you have an app running online that other people can access and use, the code running that app is considered **production code**.

TK pros and cons table for notebooks vs scripts

You'll also see many code repositories for PyTorch and ML have instructions on how to run the PyTorch code in the form of Python scripts.

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

TK image for PyTorch blog post - https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#break-down-of-key-accuracy-improvements

## TK - What we'ere going to cover

TODO

* Possible workflow: start in notebooks, explore and get something working, then go to Python scripts
* Can use "%%writefile path/to/python/script.py" as jupyter cell magic to write a Python code file 
* What should be the environment used for teaching the refactoring section? Could everyone use VSCode or Jupyter Lab? Or VS Code?
* TK - make sure the scripts have the most up to data versions of the functions in notebook 04 
* TK - how to write a code file from a notebook cell?
* TK - showcase how quickly notebook 04 can be reproduced when everything in a modular fashion (call the scripts on the command line).. e.g. `train_model.py...` -> saved model in target folder

## TK - Where can you get help?

TODO

## TK - 0. Get data ready

## TK - 1. Building the model

## TK - 2. Setting up the training code (the engine)

## TK - 3. Training the model

Pros:
* Saves a lot of rewriting code
* Good idea to: start in notebooks then move to scripts when you've got something working.

## TODO:
* Explain each of the different `.py` files
* How to go from functions in a notebook to `.py` (manaul vs automatic)

## TK - Extensions & Exercises
* TK Add an argument for using a different:
    * train/test dir
    * optimizer
    * etc...
* Use `argparse` to be able to send `train.py` custom settings for training procedures