import torch
from torch import nn

class tinyVGG(nn.Module):
  """
    Model using tinyVGG architecture

    Args:
    input_shape : int -> number of color channels
    hidden_units : int -> number of hidden units
    output_shape : int -> shape of the output
    multiplier : int -> multiplies the shape to be in the correct shape

    Functions:
    forward(x) -> returns data after going through the model

    debug_forward(x) -> returns data after going through the model and prints out the shapes after each block
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int, multiplier):
    super().__init__()
    self.block1 = nn.Sequential(
    nn.Conv2d(in_channels=input_shape,
              out_channels=hidden_units,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=hidden_units,
              out_channels=hidden_units,
              kernel_size=3,
              stride=1,
              padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,
                  stride=2)
    )
    self.block2 = nn.Sequential(
    nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,
                stride=2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features= int(hidden_units * multiplier),
                out_features=output_shape)
    )
  def forward(self, x: torch.tensor):
    return self.classifier(self.block2(self.block1(x)))

  def debug_forward(self, x: torch.tensor):
    print(f"Start shape: {x.shape}")
    x = self.block1(x)
    print(f"Shape after the first block: {x.shape}")
    x = self.block2(x)
    print(f"Shape after the second block: {x.shape}")
    x = self.classifier(x)
    print(f"Final shape: {x.shape}")
    return x


