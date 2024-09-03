import os 
import wget
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from zipfile import ZipFile
from pathlib import Path
from typing import Tuple, Dict, List, Any

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms



def init_directories(now_date_name: str) -> Tuple[str, Path, Path, Path]:

    # get current directory
    CURRENT_DIR = Path(os.getcwd())
    CURRENT_DIR_NAME = os.path.basename(CURRENT_DIR)

    # data directory
    DATA_DIR = CURRENT_DIR.joinpath("../../../data")
    if not DATA_DIR.is_dir():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # data directory
    LOG_DIR = CURRENT_DIR.joinpath(f"../../../logs/{CURRENT_DIR_NAME}_{now_date_name}")
    if not LOG_DIR.is_dir():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # data directory
    MODEL_DIR = CURRENT_DIR.joinpath(f"../../../models/{CURRENT_DIR_NAME}_{now_date_name}")
    if not MODEL_DIR.is_dir():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    return  CURRENT_DIR_NAME, DATA_DIR, LOG_DIR, MODEL_DIR


def init_file_paths(now_datetime_name: str, current_dir_name: str, log_dir: Path, model_dir: Path) -> Tuple[str, str, str, str]:

    LOG_FILE_PATH = log_dir.joinpath(f"{current_dir_name}_{now_datetime_name}.log")
    
    HISTORY_LEARNING_CURVES_FILE_PATH = log_dir.joinpath(f"{current_dir_name}_{now_datetime_name}_history_curves.png")
    
    MODEL_FILE_PATH = model_dir.joinpath(f"{current_dir_name}_{now_datetime_name}.pt")
    
    PREDICTED_RESULTS_FILE_PATH = log_dir.joinpath(f"{current_dir_name}_{now_datetime_name}_predicted_results.png")

    return  LOG_FILE_PATH, HISTORY_LEARNING_CURVES_FILE_PATH, MODEL_FILE_PATH, PREDICTED_RESULTS_FILE_PATH

def config_logging(log_file_path: Path) -> None:

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%a, %d %b %Y, %H:%M:%S",
        format="[%(asctime)s.%(msecs)03d] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file_path, mode="w"),
            logging.StreamHandler()
        ]
    )


def download_and_extract_dataset(dataset_zipfile_url: str, save_dir: Path) -> Path:

    # download dataset
    DATASET_ZIPFILE_NAME = dataset_zipfile_url.split("/")[-1]
    logging.info(f"The dataset zipfile name: {DATASET_ZIPFILE_NAME}")

    DATASET_ZIPFILE_PATH = save_dir.joinpath(DATASET_ZIPFILE_NAME)
    logging.info(f"The dataset zipfile path: {DATASET_ZIPFILE_PATH}")

    if DATASET_ZIPFILE_PATH.is_file():
        logging.info(f"The file {DATASET_ZIPFILE_NAME} already exists. Skipped downloading.")
    else:
        try:
            logging.info(f"The file {DATASET_ZIPFILE_NAME} is downloading...")
            wget.download(url=dataset_zipfile_url, out=str(save_dir))
            logging.info(f"The file {DATASET_ZIPFILE_NAME} is donwloaed successfully.")
        except Exception as error:
            logging.error(f"Caught this error: {error} during downloading!")
            return
    
    # extract dataset
    DATASET_FOLDER_NAME = DATASET_ZIPFILE_NAME.split(".")[0]
    logging.info(f"The dataset folder name: {DATASET_FOLDER_NAME}")

    DATASET_FOLDER_PATH = save_dir.joinpath(DATASET_FOLDER_NAME)
    logging.info(f"The dataset folder path: {DATASET_FOLDER_PATH}")

    if DATASET_FOLDER_PATH.is_dir():
        logging.info(f"The file {DATASET_ZIPFILE_NAME} is already extracted. Skipped extracting.")
    else:
        try:
            logging.info(f"The file {DATASET_ZIPFILE_NAME} is extracting...")
            zipper = ZipFile(file=DATASET_ZIPFILE_PATH)
            zipper.extractall(path=save_dir)
            zipper.close()
            logging.info(f"The file {DATASET_ZIPFILE_NAME} is extracted successfully.")
        except Exception as error:
            logging.error(f"Caught this error: {error} during extracting!")
            return
        
    return DATASET_FOLDER_PATH


def preprocessing_dataset(dataset_dir: Path,  
                          transform_helper: torchvision.transforms, 
                          batch_size: int = 32, 
                          num_workers: int = 4, evaluation: bool = False) ->  Tuple[DataLoader, DataLoader, List[str], int, int]:

    TRAIN_DIR = dataset_dir.joinpath("train")
    logging.info(f"Training directory: {TRAIN_DIR}")
    # walking on training set
    for dirpaths, dirnames, filenames in os.walk(TRAIN_DIR):
        if len(filenames) > 0:
            logging.info(f"There are {len(filenames)} images in {dirpaths}")

    TEST_DIR = dataset_dir.joinpath("test")
    logging.info(f"Training directory: {TEST_DIR}")
    # walking on training set
    for dirpaths, dirnames, filenames in os.walk(TEST_DIR):
        if len(filenames) > 0:
            logging.info(f"There are {len(filenames)} images in {dirpaths}")

    # train dataset
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform_helper)
    total_train_images = len(train_dataset)
    logging.info(f"There are {len(train_dataset)} images in train_dataset")

    # train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers= num_workers, pin_memory=True)

    # test dataset
    test_dataset = ImageFolder(root=TEST_DIR, transform=transform_helper)
    total_test_images = len(test_dataset)
    logging.info(f"There are {len(test_dataset)} images in test_dataset")

    # test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle = evaluation, num_workers= num_workers, pin_memory=True)

    # class names
    class_names = train_dataset.classes
    logging.info(f"Class names: {class_names}")

    return train_dataloader, test_dataloader, class_names, total_train_images, total_test_images



def create_model(device: str, num_classes: int, seeds: int = 42) -> Tuple[nn.Module, Any]:

     # get a pretrained model's weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # transforms from weights
    auto_transforms = weights.transforms()
    logging.info(f"Auto transforms from weights: {auto_transforms}")

    # freeze all base layers in the features
    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(seed=seeds)
    torch.cuda.manual_seed(seed=seeds)

    # change num class
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )

    return model, auto_transforms

def train(model: nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          epochs: int, 
          device: str, 
          loss_fn: torch.nn.Module, 
          optimizer: torch.optim, total_train_images: int, total_test_images: int) -> Tuple[Dict[str, Any], nn.Module]:

    results_dict = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }   

    for epoch in range(epochs):
        
        # accummulate history
        accummulated_train_acc = 0.0
        accummulated_train_loss = 0.0
        accummulated_train_batches = 0

        for input, output in train_dataloader:

            # training mode
            model.train()

            # send data to device
            input, output = input.to(device), output.to(device)

            # forward pass
            y_logits = model(input)

            # calculate loss
            loss = loss_fn(y_logits.float(), output.long())
            accummulated_train_loss += loss.cpu().detach().numpy()

            # calculate accuracy
            acc = torch.eq(output, torch.argmax(torch.softmax(y_logits, dim=1), dim=1)).sum().item()
            accummulated_train_acc += (acc / len(y_logits))
             
            # zero grad
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update weights
            optimizer.step()

            # update batches
            accummulated_train_batches += 1
        
        # evaluation mode
        model.eval()
        with torch.inference_mode():

            # accummulate history
            accummulated_test_acc = 0.0
            accummulated_test_acc1 = 0.0
            accummulated_test_loss = 0.0
            accummulated_test_batches = 0
            accummulate_test_length_batches = 0

            for input, output in test_dataloader:

                # send data to device
                input, output = input.to(device), output.to(device)

                # forward pass
                y_logits = model(input)

                # calculate loss
                loss = loss_fn(y_logits.float(), output.long())
                accummulated_test_loss += loss.cpu().detach().numpy()

                # calculate accuracy
                acc = torch.eq(output, torch.argmax(torch.softmax(y_logits, dim=1), dim=1)).sum().item()
                accummulated_test_acc += (acc / len(y_logits))

                # update batches
                accummulated_test_batches += 1

        train_loss = accummulated_train_loss / accummulated_train_batches
        train_accuracy = accummulated_train_acc / accummulated_train_batches
        

        val_loss = accummulated_test_loss / accummulated_test_batches
        val_accuracy = accummulated_test_acc / accummulated_test_batches
       
        results_dict["loss"].append(train_loss)
        results_dict["accuracy"].append(train_accuracy)
        results_dict["val_loss"].append(val_loss)
        results_dict["val_accuracy"].append(val_accuracy)

        logging.info(f"Epoch: {epoch + 1}/{epochs} | "
                     f"Train loss: {train_loss: .4f} | "
                     f"Train accuracy: {train_accuracy: .3f} | "
                     f"Test loss: {val_loss: .4f} | "
                     f"Test accuracy: {val_accuracy: .3f}")
        
    return results_dict, model


def save_checkpoint(model: nn.Module, file_path: Path):

    # logging.info(f"state_dict: {model.state_dict().keys()}")

    torch.save(model.state_dict(), f=file_path)
    logging.info(f"Saving a checkpoint into this path: {file_path}")

def save_learning_curves(results_dict: Dict[str, Any], file_path: Path):

    logging.info(f"Saving learning curves into this path: {file_path}")

    loss = results_dict["loss"]
    accuracy = results_dict["accuracy"]
    val_loss = results_dict["val_loss"]
    val_accuracy = results_dict["val_accuracy"]

    epochs = len(loss)
    epochs_arr = range(epochs)

    plt.figure(figsize=(10, 5))
    plt.suptitle("History of Learning Curves")
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, loss, label="Training loss") 
    plt.plot(epochs_arr, val_loss, label="Testing loss") 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, accuracy, label="Training accuracy") 
    plt.plot(epochs_arr, val_accuracy, label="Testing accuracy") 
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # save a fig
    plt.savefig(file_path, bbox_inches="tight")

def load_model(model: nn.Module, file_path: Path) -> nn.Module:
    
    logging.info(f"Loading saved state_dict from the path: {file_path}")

    try:
        model.load_state_dict(torch.load(file_path))
        logging.info(f"Loaded saved state_dict from the path: {file_path} successfully.")
    except Exception as error:
        logging.error(f"Caught this error: {error} during loading the saved state_dict.")

    return model

def evaluate(model: nn.Module, dataloader: DataLoader, device: str, num_images: int) -> Tuple[torch.Tensor, torch.Tensor, DataLoader]:

    # get one batch of images
    one_batch_images, one_batch_labels = next(iter(dataloader))
    logging.info(f"Length of a batch of images: {len(one_batch_images)}")

    # get a sub-images by num_images
    sub_images, sub_labels = None, None

    if num_images < len(one_batch_images):
        sub_images = one_batch_images[:num_images]
        sub_labels = one_batch_labels[:num_images]
    else: 
        logging.warn(f"The num_images {num_images} should be equal or less than {len(one_batch_images)}")

    # forward pass
    y_logits = model(sub_images.to(device))
    y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1).cpu().detach()

    logging.info(f"y_preds: {y_preds}")
    logging.info(f"y_truths: {sub_labels}")

    return y_preds, sub_labels, sub_images

def savefig(images: DataLoader, y_preds: torch.Tensor, y_truths: torch.Tensor , file_path: Path, class_names: List[str]) -> None:

    plt.figure(figsize=(20, 10))
    for i in range(len(images)):

        plt.subplot(2, 5, i + 1)
        
        image = images[i].permute((2,1,0)).numpy()
        plt.imshow(np.clip(image, 0., 1.0)) 
        plt.axis("off")

        label = class_names[y_truths[i]]
        prediction = class_names[y_preds[i]]

        if label == prediction:
            plt.title(f"{label} | {prediction}", color="g")
        else:
            plt.title(f"{label} | {prediction}", color="r")


    plt.savefig(file_path, bbox_inches="tight")
