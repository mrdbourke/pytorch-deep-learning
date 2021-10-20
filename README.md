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
3. PyTorch computer vision - We'll get even more specific now and see how PyTorch can be used for computer vision problems though still using the same workflow from 1 & 2.
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

## Log

Almost daily updates of what's happening.

* 20 Oct 2021 - add v0 outlines for 01 and 02, add rough outline of course to README, this course will focus on less but better 
* 19 Oct 2021 - Start repo 🔥, add fundamentals notebook draft v0
