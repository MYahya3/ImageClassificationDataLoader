import os
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transform
from PIL import Image


class FlowerDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform= None):
        self.csv_file = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.labels = self.csv_file.iloc[:,1]
        print(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.csv_file.iloc[index,0])
        image = Image.open(image_path)
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return (image, label)


a = FlowerDataset(csv_file="E:/GitHub/pycharm_projects/flowers_dataloader/flower_images/flower_labels.csv",
                  data_dir="E:/GitHub/pycharm_projects/flowers_dataloader/flower_images", transform=transform.ToTensor())

print(a[2])