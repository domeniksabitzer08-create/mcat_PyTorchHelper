from collections.abc import Callable

import torch
from torch import nn
from timeit import default_timer
from tqdm.auto import tqdm
import copy

from mcat import  evaluation


### training function ###
def train_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              acc_fn: Callable = evaluation.calculates_accuracy):
  """
      Trains the model on data_loader and returns the loss and the accuracy.
  """
  train_loss, train_acc = 0,0
  # put model on right device
  model.to(device)
  for batch, (X, y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)
    # forward pass
    y_pred = model(X)
    # calculate the loss
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += acc_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
    # optimizer zero grad
    optimizer.zero_grad()
    # loss backward
    loss.backward()
    # optimizer step
    optimizer.step()

  # Calculate the loss and accuracy
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  return train_loss, train_acc

### testing function ###
def test_step(model:torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          device: torch.device,
          acc_fn: Callable = evaluation.calculates_accuracy
          ):
  """
    Evaluates the model on data_loader and returns the loss and the accuracy
  """
  test_loss, test_acc = 0,0
  model.to(device)
  # put model to testing mode
  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X, y = X.to(device), y.to(device)
      # Forward pass
      y_pred = model(X)
      # Calculate the loss
      loss = loss_fn(y_pred, y)
      test_loss += loss
      test_acc += acc_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
    # Calculate loss and accuracy
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return test_loss, test_acc

### training and testing function ###
def train_and_test( model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader,
                    test_dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: torch.nn.Module,
                    epochs: int,
                    device: torch.device,
                    acc_fn: Callable = evaluation.calculates_accuracy,
                    ):
    """
    Train and test the model by given model and data
    returns:
            result: dictionary of train and test loss
            best_state_dict: the state_dict of the model with the lowest test loss
    """

    # Create empty results dictionary
    results = {"train_loss": [],
               "test_loss": [],
               "train_acc": [],
               "test_acc": []
              }
    lowest_loss = 1000

    # Ensure the state dict is always defined
    best_state_dict = copy.deepcopy(model.state_dict())

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                            data_loader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            acc_fn=acc_fn,
                                            device=device)

        test_loss, test_acc = test_step(model=model,
                                          data_loader=test_dataloader,
                                          loss_fn=loss_fn,
                                          acc_fn=acc_fn,
                                          device=device)

        # Print out information
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} "
            f"train_acc: {train_acc:.2f} "
            f"test_acc: {test_acc:.2f} "
        )
        # Check if it`s the best result of the model - then save the state dict
        if test_loss < lowest_loss:
          print(f"Lowest so far!")
          lowest_loss = test_loss
          best_state_dict = copy.deepcopy(model.state_dict())


        # Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results and the best state dict of the best performing model at the end of the epochs
    return results, best_state_dict