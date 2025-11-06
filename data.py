import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
import os

class GenerativeImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        
        torch.manual_seed(idx)
        img_path = os.path.join(self.img_dir, f'zero_{idx}.png')
        
        img = decode_image(img_path)
        img = img/torch.max(img)

        noise = torch.rand_like(img)
        
        return img, noise