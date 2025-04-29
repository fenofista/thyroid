import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

class Thyroid_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.data_path = "../data"
        self.df = pd.read_csv(f"{self.data_path}/{self.csv_file}")
        self.transform = transform
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ID = self.df["ID"][idx]
        dataset = self.df["dataset"][idx]

        if dataset == "DDTI":
            image_path = self.data_path+f"/DDTI dataset/DDTI/2_preprocessed_data/stage2/p_image/{ID}.PNG"
            mask_path = self.data_path+f"/DDTI dataset/DDTI/2_preprocessed_data/stage2/p_mask/{ID}.PNG"
        elif dataset == "TG3K":
            ID = str(ID).zfill(4)
            image_path = self.data_path+f"/tg3k/thyroid-image/{ID}.jpg"
            mask_path = self.data_path+f"/tg3k/thyroid-mask/{ID}.jpg"
        elif dataset == "TN3K":
            ID = str(ID).zfill(4)
            image_path = self.data_path+f"/tn3k/trainval-image/{ID}.jpg"
            mask_path = self.data_path+f"/tn3k/trainval-mask/{ID}.jpg"

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image_tensor = self.transform(image)
        mask_tensor = self.transform(mask)

            
        return image_tensor, mask_tensor