# PyTorch for Deep Learning (work in progress)
I'd like to learn PyTorch. So I'm going to use this repo to:

1. Add what I've learned.
2. Teach others in a beginner-friendly way.

Stay tuned to here for updates. Course materials will be actively worked on for the next ~3-4 months.

Launch early 2022.

## Outline

**Note:** This is rough and subject to change.

**Course focus:** code, code, code, experiment, experiment, experiment. 
**Teaching style:** [https://sive.rs/kimo](https://sive.rs/kimo)

TODO: turn the table of contents into a nice table

0. PyTorch fundamentals - ML is all about representing data as numbers (tensors) and manipulating those tensors so this module will cover PyTorch tensors.
1. PyTorch workflow - You'll use different techniques for different problem types but the workflow remains much the same:
```
data -> build model -> fit model to data (training) -> evaluate model and make predictions (inference) -> save & load model
```
Module 1 will showcase an end-to-end PyTorch workflow that can be leveraged for other problems.

2. PyTorch classification - Let's take the workflow we learned in module 1 and apply it to a common machine learning problem type: classification (deciding whether something is one thing or another).
3. PyTorch computer vision - We'll get even more specific now and see how PyTorch can be used for computer vision problems though still using the same workflow from 1 & 2. We'll also start functionizing the code we've been writing, for example: `def train(model, data, optimizer, loss_fn): ...`
4. PyTorch custom datasets - How do you load a custom dataset into PyTorch? Also we'll be laying the foundations in this notebook for our modular code (covered in 05).
5. Going modular - PyTorch is designed to be modular, let's turn what we've created into a series of Python scripts (this is how you'll often find PyTorch code in the wild). For example:
```
code/
    data_setup.py <- sets up data
    model_builder.py <- builds the model ready to be used
    engine.py <- training/eval functions for the model
    train.py <- trains and saves the model
```
6. PyTorch transfer learning - Let's improve upon the models we've built ourselves using transfer learning.
7. PyTorch experiment tracking - We've built a bunch of models... wouldn't it be good to track how they're all going?
8. ??? Milestone Project 1 & 2 could be built into the final two notebooks (putting together all the fundamentals)

As for 8, seven notebooks sounds like enough. Each will teach a maximum of 3 big ideas. 

## Status

* **Working on:** making exercises and solutions for 02
* **Next:** finish annotations for 00, 01, 02 then make keynote slides
* **Done skeleton code for:** 00, 01, 02, 03, 04, 05, 06, 07
* **Done annotations (text) for:** 00, 01, 02 
* **Done images for:**
* **Done keynotes for:** 
* **Done vidoes for:**

## TODO

See the project page for specifics - https://github.com/users/mrdbourke/projects/1 

High-level overview of things to do:
* How to use this repo (e.g. env setup, GPU/no GPU) - all notebooks should run fine in Colab and locally if needed.
* Finish skeleton code for notebooks 00 - 07 ✅
* Write annotations for 00 - 07
* Make images for 00 - 07
* Make slides for 00 - 07
* Record videos for 00 - 07

## Log

Almost daily updates of what's happening.

* 01-07 Feb 2021 - add annotations for 02, finished, still need images, going to work on exercises/solutions today 
* 31 Jan 2021 - start adding annotations for 02
* 28 Jan 2021 - add exercies and solutions for 01
* 26 Jan 2021 - lots more annotations to 01, should be finished tomorrow, will do exercises + solutions then too
* 24 Jan 2021 - add a bunch of annotations to 01
* 21 Jan 2021 - start adding annotations for 01 
* 20 Jan 2021 - finish annotations for 00 (still need to add images), add exercises and solutions for 00
* 19 Jan 2021 - add more annotations for 00
* 18 Jan 2021 - add more annotations for 00
* 17 Jan 2021 - back from holidays, adding more annotations to 00 
* 10 Dec 2021 - start adding annoations for 00
* 9 Dec 2021 - Created a website for the course ([learnpytorch.io](https://learnpytorch.io)) you'll see updates posted there as development continues 
* 8 Dec 2021 - Clean up notebook 07, starting to go back through code and add annotations
* 26 Nov 2021 - Finish skeleton code for 07, added four different experiments, need to clean up and make more straightforward
* 25 Nov 2021 - clean code for 06, add skeleton code for 07 (experiment tracking)
* 24 Nov 2021 - Update 04, 05, 06 notebooks for easier digestion and learning, each section should cover a max of 3 big ideas, 05 is now dedicated to turning notebook code into modular code 
* 22 Nov 2021 - Update 04 train and test functions to make more straightforward
* 19 Nov 2021 - Added 05 (transfer learning) notebook, update custom data loading code in 04
* 18 Nov 2021 - Updated vision code for 03 and added custom dataset loading code in 04
* 12 Nov 2021 - Added a bunch of skeleton code to notebook 04 for custom dataset loading, next is modelling with custom data
* 10 Nov 2021 - researching best practice for custom datasets for 04
* 9 Nov 2021 - Update 03 skeleton code to finish off building CNN model, onto 04 for loading custom datasets
* 4 Nov 2021 - Add GPU code to 03 + train/test loops + `helper_functions.py`
* 3 Nov 2021 - Add basic start for 03, going to finish by end of week
* 29 Oct 2021 - Tidied up skeleton code for 02, still a few more things to clean/tidy, created 03
* 28 Oct 2021 - Finished skeleton code for 02, going to clean/tidy tomorrow, 03 next week
* 27 Oct 2021 - add a bunch of code for 02, going to finish tomorrow/by end of week
* 26 Oct 2021 - update 00, 01, 02 with outline/code, skeleton code for 00 & 01 done, 02 next
* 23, 24 Oct 2021 - update 00 and 01 notebooks with more outline/code
* 20 Oct 2021 - add v0 outlines for 01 and 02, add rough outline of course to README, this course will focus on less but better 
* 19 Oct 2021 - Start repo 🔥, add fundamentals notebook draft v0
