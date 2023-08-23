import torch
import os
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, transforms):
    ### data_root - path to dir containing images (unlabelled)
    ### transforms - for images
        self.data_root  = data_root
        self.transforms = transforms

        self.img_names  = os.listdir(data_root)
        self.length     = len(self.img_names)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img     = Image.open(os.path.join(self.data_root, self.img_names[idx]))
        return [self.transforms(img)]

