import torch

import torch.nn as nn
import torchvision
import torchinfo

import os 
import wget
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from zipfile import ZipFile
from pathlib import Path

import helper_functions

def main():

    # datetime
    now_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_date_name = datetime.now().strftime("%Y%m%d")
    now_datetime_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--evaluation", default=True, type=bool)
    parser.add_argument("--seeds", default=42, type=int)

    # args 
    args = parser.parse_args()

    # init directories
    CURRENT_DIR_NAME, DATA_DIR, LOG_DIR, MODEL_DIR = helper_functions.init_directories(now_date_name=now_date_name)

    # init files 
    LOG_FILE_PATH, HISTORY_LEARNING_CURVES_FILE_PATH, MODEL_FILE_PATH, PREDICTED_RESULTS_FILE_PATH =  helper_functions.init_file_paths(now_datetime_name = now_datetime_name, 
                                     current_dir_name = CURRENT_DIR_NAME, 
                                     log_dir = LOG_DIR, 
                                     model_dir = MODEL_DIR)
    
    # config logging
    helper_functions.config_logging(log_file_path=LOG_FILE_PATH)

    # library versions
    logging.info(f"Torch version: {torch.__version__}")
    logging.info(f"Torchvision version: {torchvision.__version__}")

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    logging.info(f"Cuda version: {torch.version.cuda}")

    # num_cpus
    num_workers = os.cpu_count()

    # hyper-parameters
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Batch size: {args.batch_size}")

    logging.info(f"Evalution mode: {args.evaluation}")
    logging.info(f"Training mode: {not args.evaluation}")

    # download dataset
    DATASET_ZIPFILE_URL = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    DATASET_FOLDER_PATH = helper_functions.download_and_extract_dataset(dataset_zipfile_url = DATASET_ZIPFILE_URL, 
                                      save_dir = DATA_DIR)
    
    # create model
    model, auto_transforms = helper_functions.create_model(device=device, num_classes=3, seeds = args.seeds)
    logging.info(torchinfo.summary(model=model, input_size=(args.batch_size, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"]))


    # preprocessing dataset
    train_dataloader, test_dataloader, class_names, total_train_images, total_test_images = helper_functions.preprocessing_dataset(dataset_dir = DATASET_FOLDER_PATH, 
                                           transform_helper= auto_transforms,
                                           batch_size=args.batch_size,
                                           num_workers=num_workers,
                                           evaluation = args.evaluation)
     
    # loss_fn 
    loss_fn = torch.nn.CrossEntropyLoss()

    # opimizer
    optimizer = torch.optim.Adam(params= model.parameters(), lr=args.learning_rate)

    # training 
    if not args.evaluation:
        # train model
        results_dict, model = helper_functions.train(model=model, 
                                            train_dataloader=train_dataloader,
                                            test_dataloader= test_dataloader, 
                                            epochs= args.epochs,
                                            device=device, 
                                            loss_fn=loss_fn, 
                                            optimizer= optimizer, total_test_images=total_test_images, total_train_images = total_train_images)
        # save learning curves
        helper_functions.save_learning_curves(results_dict=results_dict, file_path=HISTORY_LEARNING_CURVES_FILE_PATH)
        
        # save models
        helper_functions.save_checkpoint(model=model, file_path = MODEL_FILE_PATH)
    else: 
        
        # get the model file path
        MODEL_FILE_PATH = next(MODEL_DIR.glob("*.pt"))

        # create an instance of the model
        new_model, _ = helper_functions.create_model(device=device, num_classes=len(class_names))

        # load the saved state_dict of the model by its path
        loaded_model = helper_functions.load_model(model=new_model, file_path = MODEL_FILE_PATH).to(device)

        # evaluation
        y_preds, y_truths, images = helper_functions.evaluate(model=loaded_model, dataloader = test_dataloader, device=device,  num_images = 10)

        # save predictions 
        helper_functions.savefig(images = images, y_preds = y_preds, y_truths = y_truths, file_path= PREDICTED_RESULTS_FILE_PATH, class_names=class_names)

if __name__ == "__main__":
    main()