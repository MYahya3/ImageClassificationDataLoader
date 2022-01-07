import torch
from torch.utils.data import DataLoader
from MyDataset import FlowerDataset
import torch.nn as nn
import torch.nn.functional as F
from MyDataset import FlowerDataset

import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
  for epoch in range(1, n_epochs + 1):
      loss_train = 0.0

      for imgs, labels in train_loader:
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

      if epoch == 1 or epoch % 2 == 1:
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))

# Build Model class

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 12, kernel_size=5,padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(12, 8, kernel_size=5, padding=0)
        self.fcl1 = nn.Linear(29*29*8, 120)
        self.fcl2 = nn.Linear(120, 84)
        self.fcl3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        return x

import torchvision.transforms as transforms

dataset = FlowerDataset(csv_file="E:/GitHub/pycharm_projects/flowers_dataloader/flower_images/flower_labels.csv",
               data_dir="E:/GitHub/pycharm_projects/flowers_dataloader/flower_images", transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(128)]))

loader = DataLoader(dataset, batch_size=4)

import torch.optim as optim
model = model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training_loop(n_epochs=20, optimizer=optimizer, model=model, loss_fn=criterion, train_loader= loader)

