import torch, torchvision
from torch.utils.data import DataLoader
from text_format_dataloader import TextDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform
import os
path = "E:/GitHub/pycharm_projects/data_loader_with_text_format"
transform = transform.Compose([transform.Resize(224),transform.ToTensor()])
dataset = TextDataset(path, transform)
loader = DataLoader(dataset, batch_size=1)

# print(dataset[0])

# pre-trained Model
model = torchvision.models.resnet50()

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)
# model.to(device)

import torch.optim as optim
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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
        # print(loss_train)
      print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))

training_loop(n_epochs=20, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader= loader)




