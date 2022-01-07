import torchvision.transforms as transform
from torch.utils.data import Dataset
import glob
from PIL import Image

class TextDataset(Dataset):
    def __init__(self, path, transform= None):
        self.images_list = glob.glob(path + '/*/*.jpg')
        self.labels_file = glob.glob(path + "/*/*.txt")
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        labels = []
        for label in self.labels_file:
            with open(label) as f:
                print(label)# open the file and then call .read() to get the text
                labels.append([int(i) for i in (f.read(1))])

        label_id = labels[index][0]
        image = Image.open(self.images_list[index])
        image = self.transform(image)
        return image, label_id

# Example
path = "E:/GitHub/pycharm_projects/data_loader_with_text_format"
transform = transform.ToTensor()
a = TextDataset(path, transform)
print(a[0])
print(len(a))