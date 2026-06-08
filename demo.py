import splitfolders
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR as EXplr
import wandb
#from Final import model
from Final import ConvNet, numchange

image_path = "K.png"
img = Image.open(image_path)


val_test_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

post_augmentation_img = val_test_transforms(img)

image_display = post_augmentation_img.permute(1, 2, 0)

plt.figure()
plt.imshow(image_display)
plt.axis('off')
plt.title("Post-augmentation")
plt.show()

model = ConvNet()

model.load_state_dict(torch.load("model.pt", weights_only=True))

model.eval()

demo_input = post_augmentation_img.unsqueeze(0)

# Run through model
with torch.no_grad():  
    output = model(demo_input)

probs = torch.softmax(output, dim=1)

predicted_class = torch.argmax(probs, dim=1).item()
predicted_letter = numchange(predicted_class)

print("\nAll probabilities:")
for i in range(28):
    print(f"{numchange(i)}: {probs[0][i].item() * 100:.1f}%")

print(predicted_letter)


# this line is used to load the trained weights into a new file