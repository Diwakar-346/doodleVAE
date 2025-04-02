# src/data_loader.py
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class DoodlesDataset(Dataset):
    def __init__(self, npy_file, transform=None):
        self.data = np.load(npy_file)
        # normalize t0 [0,1] for sigmoid
        self.data = self.data.astype('float32') / 255.0
        self.data = self.data.reshape(-1, 28, 28)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomAffine(degrees=10, shear=5),
    transforms.RandomHorizontalFlip(),
])

def get_data_loaders(dataset, train_split_ratio, batch_size, transform=transform):
    dataset = DoodlesDataset(dataset, transform)
    total_size = len(dataset)
    train_size = int(train_split_ratio * total_size)
    test_size = total_size - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader