import torch
import torchvision
from torchvision import datasets, transforms

def predict_on_custom_data(img_path: str, model: torch.nn.Module, device, size: tuple):
  """
  Loads in a custom image and predicts on it

  :param img_path: path of the images the model will predict on
  :param model: model that performs the prediction
  :param device: device of the model
  :param size: size of the image for the model
  :return:
  prediction label index
  prediction probability
  """

  # Transform the img:
  # Load in custom image and convert it
  img_raw = torchvision.io.read_image(img_path)
  img = img_raw.type(torch.float32) /255.
  # Put it on the right device
  img = img.to(device)
  # Put the image in the right size
  img_transform = transforms.Compose([
     transforms.Resize(size=size)
  ])
  img = img_transform(img)
  # Add a batch size
  img = img.unsqueeze(1)
  # Change the order
  img = img.permute(1,0,2,3)
  #Make prediction
  model.eval()
  with torch.inference_mode():
    img_pred = model(img)
  # logits -> probabilities
  img_pred = torch.softmax(img_pred, dim=1)
  # probabilities -> labels
  img_label = torch.argmax(img_pred)
  # Turn the prediction into %-probability
  img_pred_percent =  img_pred.squeeze(dim=0)[img_label] * 100
  return img_label, img_pred_percent
