import torch
from tqdm import tqdm  # Standard usage for scripts
import numpy as np
from typing import Tuple, Dict, List
import random
import matplotlib.pyplot as plt

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device="cpu"):
    
    """Performs the training step of a neural network.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        Sumary of train loss and train accuracy.
    """

    train_loss, train_acc = 0, 0
    model.to(device)
    model.train() # put model in train mode
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,              
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device="cpu"):
    
    """Performs the test step of a neural network.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        Sumary of train loss and train accuracy.
    """

    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
         
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def train_nn(model: torch.nn.Module,
             train_data_loader: torch.utils.data.DataLoader,
             test_data_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             accuracy_fn,
             device="cpu",
             epochs=10):
    
    """train_step and test_step in one single function.
    """

   
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train() # put model in train mode

    for epoch in tqdm(range(epochs)):
        
        ### Training

        print(f"Epoch: {epoch}\n---------")
        for batch, (X, y) in enumerate(train_data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_true=y,
                                    y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        # Calculate loss and accuracy per epoch and print out what's happening
        train_loss /= len(train_data_loader)
        train_acc /= len(train_data_loader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

        ### Testing

        test_loss, test_acc = 0, 0
        model.to(device)
        model.eval() # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode(): 
            for X, y in test_data_loader:
                # Send data to GPU
                X, y = X.to(device), y.to(device)
                
                # 1. Forward pass
                test_pred = model(X)
                
                # 2. Calculate loss and accuracy
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y,
                    y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
                )
            
            # Adjust metrics and print out
            test_loss /= len(test_data_loader)
            test_acc /= len(test_data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn, 
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    # model.to(device)  may not be necessary
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


def make_predictions(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device="cpu"):
    y_preds = []
    model.eval()
    model.to(device)
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):

            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            
            # Do the forward pass
            y_logit = model(X)

            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())

    # Concatenate list of predictions into a tensor
    return torch.cat(y_preds)


def display_random_images(dataset: torch.utils.data.dataset.Dataset, # or torchvision.datasets.ImageFolder?
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          rows: int = 5,
                          cols: int = 5,
                          seed: int = None):
    
   
    """Displays a number of random images from a given dataset.

    Args:
        dataset (torch.utils.data.dataset.Dataset): Dataset to select random images from.
        classes (List[str], optional): Names of the classes. Defaults to None.
        n (int, optional): Number of images to display. Defaults to 10.
        display_shape (bool, optional): Whether to display the shape of the image tensors. Defaults to True.
        rows: number of rows of the subplot
        cols: number of columns of the subplot
        seed (int, optional): The seed to set before drawing random images. Defaults to None.
    
    Usage:
    display_random_images(train_data, 
                      n=16, 
                      classes=class_names,
                      rows=4,
                      cols=4,
                      display_shape=False,
                      seed=None)
    """

    # 1. Setup the range to select images
    n = min(n, len(dataset))
    # 2. Adjust display if n too high
    if n > rows*cols:
        n = rows*cols
        #display_shape = False
        print(f"For display purposes, n shouldn't be larger than {rows*cols}, setting to {n} and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(cols*4, rows*4))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(rows, cols, i+1)        
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)