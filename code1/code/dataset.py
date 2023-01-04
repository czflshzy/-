# train : 0.png ~ 2999.png
# test : 3000.png ~ 4083.png
import glob

from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms as T


class BitmojiDataset(Dataset):
    def __init__(self, folder_path, lists, idx_list, transforms=None, train=False, valid=False, test=False):
        self.path = folder_path
        self.data_len = len(idx_list)
        self.train = train
        self.valid = valid
        self.test = test

        if train or valid:
            with open(lists, 'r') as f:
                lines = f.readlines()
        imgs = []
        labels = []
        keyPoints = []
        if train or valid:
            first = 0
            for line in lines:
                if first == 1:
                    line = line.strip()
                    s_list = line.split(',')
                    if int(s_list[0][:-4]) in idx_list:
                        imgs.append(s_list[0])
                        if int(s_list[1]) == -1:
                            labels.append(0)
                        else:
                            labels.append(1)
                        keyPoints.append([int(s) for s in s_list[2:]])
                elif first == 0:
                    first = 1
        else:
            imgs = [f"{i}.jpg" for i in idx_list]
        self.imgs = imgs
        self.labels = labels
        self.keyPoints = keyPoints

        if transforms is None:
            self.transforms = T.Compose([
                T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
                T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.path + self.imgs[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        if self.train or self.valid:
            label = self.labels[index]
            return data, label
        elif self.test:
            return self.imgs[index], data

    def __len__(self):
        return self.data_len
