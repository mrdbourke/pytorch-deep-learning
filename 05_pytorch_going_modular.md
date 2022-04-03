# TODO: Going modular

* TK - make sure the scripts have the most up to data versions of the functions in notebook 04 
* TK - showcase how quickly notebook 04 can be reproduced when everything in a modular fashion (call the scripts on the command line).. e.g. `train_model.py...` -> saved model in target folder

In this section we turn notebook 04 into a series of Python scripts saved to `going_modular`.

This document will describe those steps.

Afterwards we can reuse the same code in the scripts in the next notebook (06), Transfer Learning.

Pros:
* Saves a lot of rewriting code
* Good idea to: start in notebooks then move to scripts when you've got something working.

## TODO:
* Explain each of the different `.py` files
* How to go from functions in a notebook to `.py` (manaul vs automatic)

## Extensions & Exercises
* Use `argparse` to be able to send `train.py` custom settings for training procedures