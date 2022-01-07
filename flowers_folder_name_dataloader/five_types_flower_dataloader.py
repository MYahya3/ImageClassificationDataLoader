from torch.utils.data import Dataset
import glob
import torchvision
import torchvision.transforms as transforms
from PIL import Image

                                        ## Five_flowers_type_Dataset
class FiveFlowersDataset(Dataset):
    def __init__(self, path, labels_map, transform=None):
        self.files_dir = glob.glob(path + "/*/*")
        self.labels = [filepath.split('\\')[-2] for filepath in self.files_dir]
        self.labels_map = labels_map    # Convert (string labels into int)
        self.transform = transform
    # To find len
    def __len__(self):
        return len(self.files_dir)

    def __getitem__(self, index):
        label_id = self.labels_map[self.labels[index]]
        file = self.files_dir[index]
        image = Image.open(file)
        image = self.transform(image)
        return (image, label_id)


# Example
label_map = {"bellflower":0,"pot marigold":1, "rosy_evening":2, "Silver_slipper_orchid":3, "tuberous_pea":4}
path = "E:/GitHub/pycharm_projects/5_type_flower_dataloader/train"
transform = torchvision.transforms.Compose([transforms.ToTensor()])
a = FiveFlowersDataset(path,label_map,transform=transform)
print(a[2])