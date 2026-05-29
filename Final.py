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

# split the folders into 3 datasets training validation and testing being split 70, 15, 15% respectivley
#og_folder = "Type_01_Raw_Gesture"
#splitfolders.ratio(og_folder, output="asl_folder", seed=76, ratio=(.7,.15,.15))



#data augemntations for data
augmentations = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)),
    #v2.RandomResizedCrop(size = (224,224), scale = (0.08,1.0), ratio = (0.75, 1.33)),
    v2.RandomHorizontalFlip(0.25),
    v2.RandomPerspective(p =0.3),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('asl_folder/train', transform=augmentations)
val_dataset = datasets.ImageFolder('asl_folder/val', transform=val_test_transforms)
test_dataset = datasets.ImageFolder('asl_folder/test', transform=val_test_transforms)

#data loaders for the dataset
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True) # Add num_workers=16
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True) # Add num_workers=16
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True) # Add num_workers=16

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

#when visualizing data make it so that when displaying show the correct letter to image
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
    

#visualization code for the images
for num, image in enumerate(images):
    image = image.permute(1,2,0)
    plt1 = plt.subplot(10, 10, num+1)
    plt1.imshow(image)
    plt1.set_title(numchange(labels[num]))
    plt1.axis('off')
plt.show()


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 48, 3, 1, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(28 * 28 * 48, 400)
        self.fc2 = nn.Linear(400, 28)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output


model = ConvNet()
model.train()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 20


# -------TRAINING LOOP-------------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch1, batch2 in train_loader:
        train_preds = model(batch1)
        loss = criterion(train_preds, batch2.unsqueeze(1))
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch:{epoch} | Loss {total_loss/len(train_loader) ** .5}")

# -------VALIDATION LOOP-------------
    model.eval()
    total_loss1 = 0
    for val_x, val_y in val_loader:
        val_preds = model(val_x)
        loss1 = criterion(val_preds, val_y.unsqueeze(1))
        total_loss1 += loss1
    print(f"Epoch {epoch} | Loss: {(total_loss1/(len(val_loader)))**0.5}")


# -------TESTING LOOP-------------
model.eval()
with torch.no_grad():
    ### Get inputs and outputs in batches using the testing DataLoader
    total_loss2 = 0
    for test_x, test_y in test_loader:
        test_preds = model(test_x)
        loss2 = criterion(test_preds, test_y.unsqueeze(1))
        total_loss2 += loss2

    print(f"Loss: {(total_loss2/(len(test_loader)))**0.5}")      
# no dataset
