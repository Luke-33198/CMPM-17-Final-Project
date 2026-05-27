import splitfolders
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torchvision.transforms import v2

og_folder = "Type_01_(Raw_Gesture)"
splitfolders.ratio(og_folder, output="asl_folder", seed=76, ratio=(.7,.15,.15))

train_dataset = datasets.ImageFolder('asl_folder/train', transform=transforms.ToTensor())
val_dataset = datasets.ImageFolder('asl_folder/val', transform=transforms.ToTensor())
test_dataset = datasets.ImageFolder('asl_folder/test', transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

augmentations = v2.Compose([
    v2.ToTensor(),
    v2.RandomResizedCrop(size = (224,224), scale = (0.08,1.0), ratio = (0.75, 1.33)),
    v2.RandomHorizontalFlip(0.25),
    v2.RandomPerspective(p =0.3),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for images, labels in train_loader:
    #This where we create augmentations 
    images = augmentations(images)
    break

for images_val, labels_val in val_loader:
    images_val = val_test_transforms(images_val)
    break

for images_test, labels_test in test_loader:
    images_test = val_test_transforms(images_test)
    break
fig = plt.figure(figsize=(40,40))


def numchange(num):
    match num:
        case 0:
            return "A"
        case 1:
            return "B"
        case 2:
            return "C"
        case 3:
            return "D"
        case 4:
            return "DEL"
        case 5:
            return "E"
        case 6:
            return "F"
        case 7:
            return "G"
        case 8:
            return "H"
        case 9:
            return "I"
        case 10:
            return "J"
        case 11:
            return "K"
        case 12:
            return "L"
        case 13:
            return "M"
        case 14:
            return "N"
        case 15:
            return "O"
        case 16:
            return "P"
        case 17:
            return "Q"
        case 18:
            return "R"
        case 19:
            return "S"
        case 20:
            return "SPACE"
        case 21:
            return "T"
        case 22:
            return "U"
        case 23:
            return "V"
        case 24:
            return "W"
        case 25:
            return "X"
        case 26:
            return "Y"
        case 27:
            return "Z"
    


for num, image in enumerate(images):
    image = image.permute(1,2,0)
    plt1 = plt.subplot(10, 10, num+1)
    plt1.imshow(image)
    plt1.set_title(numchange(labels[num]))
    plt1.axis('off')
plt.show()

class myModel(nn.Module):
    def__init__self


