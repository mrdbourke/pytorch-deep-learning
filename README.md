# Learn PyTorch for Deep Learning (work in progress)

I'd like to learn PyTorch. So I'm going to use this repo to:

1. Add what I've learned.
2. Teach others in a beginner-friendly way.

Stay tuned to here for updates, course materials are being actively worked on.

Launch early-mid 2022.

## Course materials/outline

* **Note:** This is rough and subject to change.
* **Course focus:** `code, code, code, experiment, experiment, experiment`
* **Teaching style:** [https://sive.rs/kimo](https://sive.rs/kimo)

| **Section** | **What does it cover?** | **Exercises & Extra-curriculum** | **Slides** |
| ----- | ----- | ----- | ----- |
| [00 - PyTorch Fundamentals](https://www.learnpytorch.io/00_pytorch_fundamentals/) | Many fundamental PyTorch operations used for deep learning and neural networks. | [Go to exercises & extra-curriculum](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises) | [Go to slides](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/00_pytorch_and_deep_learning_fundamentals.pdf) |
| [01 - PyTorch Workflow](https://www.learnpytorch.io/01_pytorch_workflow/) | Provides an outline for approaching deep learning problems and building neural networks with PyTorch. | [Go to exercises & extra-curriculum](https://www.learnpytorch.io/01_pytorch_workflow/#exercises) | [Go to slides](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/01_pytorch_workflow.pdf) |
| [02 - PyTorch Neural Network Classification](https://www.learnpytorch.io/02_pytorch_classification/) | Uses the PyTorch workflow from 01 to go through a neural network classification problem. | [Go to exercises & extra-curriculum](https://www.learnpytorch.io/02_pytorch_classification/#exercises) | [Go to slides](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/02_pytorch_classification.pdf) |
| [03 - PyTorch Computer Vision](https://www.learnpytorch.io/03_pytorch_computer_vision/) | Let's see how PyTorch can be used for computer vision problems using the same workflow from 01 & 02. | [Go to exercises & extra-curriculum](https://www.learnpytorch.io/03_pytorch_computer_vision/#exercises) | [Go to slides](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/03_pytorch_computer_vision.pdf) |
| [04 - PyTorch Custom Datasets](https://www.learnpytorch.io/04_pytorch_custom_datasets/) | How do you load a custom dataset into PyTorch? Also we'll be laying the foundations in this notebook for our modular code (covered in 05). | [Go to exercises & extra-curriculum](https://www.learnpytorch.io/04_pytorch_custom_datasets/#exercises) | Go to slides |
| Coming soon: 05 - Going modular | PyTorch is designed to be modular, let's turn what we've created into a series of Python scripts (this is how you'll often find PyTorch code in the wild). | Go to exercises & extra-curriculum | Go to slides |
| Coming soon: 06 - PyTorch Transfer Learning | Let's take a well performing pre-trained model and adjust it to one of our own problems. | Go to exercises & extra-curriculum | Go to slides |
| Coming soon: 07 - Milestone Project 1: PyTorch Experiment Tracking | We've built a bunch of models... wouldn't it be good to track how they're all going? | Go to exercises & extra-curriculum | Go to slides |
| Coming soon: 08 - Milestone Project 2: PyTorch Paper Replicating | PyTorch is the most popular deep learning framework for machine learning research, let's see why by replicating a machine learning paper. | Go to exercises & extra-curriculum | Go to slides |
| Coming soon: 09 - Milestone Project 3: Model deployment | So you've built a working PyTorch model... how do you get it in the hands of others? Hint: deploy it to the internet. | Go to exercises & extra-curriculum | Go to slides |

### Old outline version (will update this if necessary)

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
8. PyTorch paper replicating - Let's see why PyTorch is the most popular deep learning framework for machine learning research by replicating a machine learning research paper with it.
9. PyTorch model deployment - How do you get your PyTorch models in the hands of others?

Each notebook will teach a maximum of 3 big ideas. 

## Status

* **Working on:** record course videos for notebook 04
* **Total video count:** 125
* **Next:** make materials for section 05
* **Done skeleton code for:** 00, 01, 02, 03, 04, 05, 06, 07
* **Done annotations (text) for:** 00, 01, 02, 03, 04
* **Done images for:** 00, 01, 02, 03, 04
* **Done keynotes for:** 00, 01, 02, 03
* **Done exercises and solutions for:** 00, 01, 02, 03, 04
* **Done vidoes for:** 00, 01, 02, 03

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

* 23 Apr 2022 - recorded 6 videos for 03, finished videos for 03, now to 04 
* 22 Apr 2022 - recorded 5 videos for 03
* 21 Apr 2022 - recorded 9 videos for 03
* 20 Apr 2022 - recorded 3 videos for 03
* 19 Apr 2022 - recorded 11 videos for 03
* 18 Apr 2022 - finish exercises/solutions for 04, added live-coding walkthrough of 04 exercises/solutions on YouTube: https://youtu.be/vsFMF9wqWx0
* 16 Apr 2022 - finish exercises/solutions for 03, added live-coding walkthrough of 03 exercises/solutions on YouTube: https://youtu.be/_PibmqpEyhA
* 14 Apr 2022 - add final images/annotations for 04, begin on exercises/solutions for 03 & 04
* 13 Apr 2022 - add more images/annotations for 04
* 3 Apr 2022 - add more annotations for 04
* 2 Apr 2022 - add more annotations for 04
* 1 Apr 2022 - add more annotations for 04
* 31 Mar 2022 - add more annotations for 04
* 29 Mar 2022 - add more annotations for 04
* 27 Mar 2022 - starting to add annotations for 04
* 26 Mar 2022 - making dataset for 04
* 25 Mar 2022 - make slides for 03
* 24 Mar 2022 - fix error for 03 not working in docs (finally)
* 23 Mar 2022 - add more images for 03
* 22 Mar 2022 - add images for 03
* 20 Mar 2022 - add more annotations for 03
* 18 Mar 2022 - add more annotations for 03
* 17 Mar 2022 - add more annotations for 03 
* 16 Mar 2022 - add more annotations for 03
* 15 Mar 2022 - add more annotations for 03
* 14 Mar 2022 - start adding annotations for notebook 03, see the work in progress here: https://www.learnpytorch.io/03_pytorch_computer_vision/
* 12 Mar 2022 - recorded 12 videos for 02, finished section 02, now onto making materials for 03, 04, 05
* 11 Mar 2022 - recorded 9 videos for 02
* 10 Mar 2022 - recorded 10 videos for 02
* 9 Mar 2022 - cleaning up slides/code for 02, getting ready for recording
* 8 Mar 2022 - recorded 9 videos for section 01, finished section 01, now onto 02
* 7 Mar 2022 - recorded 4 videos for section 01
* 6 Mar 2022 - recorded 4 videos for section 01
* 4 Mar 2022 - recorded 10 videos for section 01
* 20 Feb 2022 - recorded 8 videos for section 00, finished section, now onto 01
* 18 Feb 2022 - recorded 13 videos for section 00
* 17 Feb 2022 - recorded 11 videos for section 00 
* 16 Feb 2022 - added setup guide 
* 12 Feb 2022 - tidy up README with table of course materials, finish images and slides for 01
* 10 Feb 2022 - finished slides and images for 00, notebook is ready for publishing: https://www.learnpytorch.io/00_pytorch_fundamentals/
* 01-07 Feb 2022 - add annotations for 02, finished, still need images, going to work on exercises/solutions today 
* 31 Jan 2022 - start adding annotations for 02
* 28 Jan 2022 - add exercies and solutions for 01
* 26 Jan 2022 - lots more annotations to 01, should be finished tomorrow, will do exercises + solutions then too
* 24 Jan 2022 - add a bunch of annotations to 01
* 21 Jan 2022 - start adding annotations for 01 
* 20 Jan 2022 - finish annotations for 00 (still need to add images), add exercises and solutions for 00
* 19 Jan 2022 - add more annotations for 00
* 18 Jan 2022 - add more annotations for 00
* 17 Jan 2022 - back from holidays, adding more annotations to 00 
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
