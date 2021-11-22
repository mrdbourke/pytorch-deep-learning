# pytorch-deep-learing (work in progress)
I'd like to learn PyTorch. So I'm going to use this repo to:

1. Add what I've learned.
2. Teach others in a beginner-friendly way.

Stay tuned to here for updates. Course materials will be actively worked on for the next ~3-4 months.

Launch early 2022.

## Outline

**Note:** This is rough and subject to change.

**Course focus:** code, code, code, experiment, experiment, experiment. 
**Teaching style:** https://sive.rs/kimo

0. PyTorch fundamentals - ML is all about representing data as numbers (tensors) and manipulating those tensors so this module will cover PyTorch tensors.
1. PyTorch workflow - You'll use different techniques for different problem types but the workflow remains much the same:
```
data -> build model -> fit model to data (training) -> evaluate model and make predictions (inference) -> save & load model
```
Module 1 will showcase an end-to-end PyTorch workflow that can be leveraged for other problems.

2. PyTorch classification - Let's take the workflow we learned in module 1 and apply it to a common machine learning problem type: classification (deciding whether something is one thing or another).
3. PyTorch computer vision - We'll get even more specific now and see how PyTorch can be used for computer vision problems though still using the same workflow from 1 & 2. We'll also start functionizing the code we've been writing, for example: `def train(model, data, optimizer, loss_fn): ...`
4. Going modular - PyTorch is designed to be modular, let's turn what we've created into a series of Python scripts (this is how you'll often find PyTorch code in the wild). For example:
```
code/
    model.py
    training.py
    eval.py
```
5. ??? 

Still tossing up ideas for the last one. Possibly a two scaled up projects to emphasize everything in 2, 3, 4.

Some ideas: transfer learning + replicate a modern paper with pure PyTorch?

## Status

**Working on:** skeleton code for 03
**Next:** Finished skeleton code for 03 then start on 04 (computer vision with custom datasets)
**Done skeleton code for:** 00, 01, 02 

## TODO

High-level overview of things to do:
* How to use this repo (e.g. env setup, GPU/no GPU) - all notebooks should run fine in Colab and locally if needed.
* Finish skeleton code for notebooks 00 - 05
* Make slides for 00 - 05
* Write annotations for 00 - 05
* Record videos for 00 - 05

## Log

Almost daily updates of what's happening.

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
