import os
import cv2
import torch
import torch.utils.data as tud
import torchvision.transforms as trans


class ImageFolderDataset(tud.Dataset):
    def __init__(self, root, image_size, mode='TRAIN'):
        super(ImageFolderDataset, self).__init__()
        self.image_size = image_size
        if mode == 'TRAIN':
            root = root + '/train'
        else:
            root = root + '/val'
        self.data = []
        self.classes = os.listdir(root)
        for target in range(len(self.classes)):
            data_root = root + '/' + self.classes[target]
            for data in os.listdir(data_root):
                self.data.append([data, target])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, class_index = self.data[index]

        img = cv2.imread(image_path, 1)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.CV_32FC3, 1 / 255.0)
        tensor_img = trans.ToTensor()(img)

        return tensor_img, torch.Tensor(class_index)
