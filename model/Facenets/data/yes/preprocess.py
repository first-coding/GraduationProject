import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, data_dir, image_size=(160, 160)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_paths = []
        self.labels = []
        
        # 遍历数据目录中的每个子文件夹
        for label_folder in os.listdir(data_dir):
            label_folder_path = os.path.join(data_dir, label_folder)
            if os.path.isdir(label_folder_path):
                for image_file in os.listdir(label_folder_path):
                    image_path = os.path.join(label_folder_path, image_file)
                    if image_path.endswith('.jpg') or image_path.endswith('.png'):
                        self.image_paths.append(image_path)
                        self.labels.append(label_folder)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        label = int(self.labels[idx])  # 假设标签为文件夹名，已经编码为整数
        return image, label

def load_data(data_dir, batch_size=32):
    dataset = FaceDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
