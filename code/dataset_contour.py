import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tx
import random
import cv2
from PIL import ImageEnhance
from skimage.exposure import match_histograms
import torchvision.transforms as T
import torchvision.transforms.functional as F
class Thyroid_Dataset(Dataset):
    def __init__(self, csv_file, transform, image_size = 128, return_from_dataset = False, crop_DDTI = False, histo_match = False):
        self.csv_file = csv_file
        self.data_path = "../data"
        self.transform = transform
        self.df = pd.read_csv(f"{self.data_path}/{self.csv_file}")
        self.cache = {}
        self.return_from_dataset = return_from_dataset
        self.crop_DDTI = crop_DDTI
        self.histo_match = histo_match
        image_path = self.data_path+f"/tn3k/trainval-image/0001.jpg"
        self.histo_match_image = Image.open(image_path).convert("L")
        self.image_size = image_size
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ID = self.df["ID"][idx]
        dataset = self.df["dataset"][idx]
        
        
        if f"{dataset}_{ID}" in self.cache:
            image, mask, seg_type, from_dataset = self.cache[f"{dataset}_{ID}"]
        else:
            # seg_type : 1(nodule), 2(gland)
            if dataset == "DDTI":
                image_path = self.data_path+f"/DDTI dataset/DDTI/1_or_data/image/{ID}.PNG"
                mask_path = self.data_path+f"/DDTI dataset/DDTI/1_or_data/mask/{ID}.PNG"
                # image_path = self.data_path+f"/DDTI dataset/DDTI/2_preprocessed_data/stage2/p_image/{ID}.PNG"
                # mask_path = self.data_path+f"/DDTI dataset/DDTI/2_preprocessed_data/stage2/p_mask/{ID}.PNG"
                seg_type = 1
                from_dataset = 1
            elif dataset == "TG3K":
                ID = str(ID).zfill(4)
                image_path = self.data_path+f"/tg3k/thyroid-image/{ID}.jpg"
                mask_path = self.data_path+f"/tg3k/thyroid-mask/{ID}.jpg"
                seg_type = 2
                from_dataset = 2
            elif dataset == "TN3K":
                ID = str(ID).zfill(4)
                image_path = self.data_path+f"/tn3k/trainval-image/{ID}.jpg"
                mask_path = self.data_path+f"/tn3k/trainval-mask/{ID}.jpg"
                seg_type = 1
                from_dataset = 3
    
            image = Image.open(image_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            if dataset == "DDTI" and self.crop_DDTI:
                # print("before crop : ", image.size)
                image, mask = self.crop_ddti_ultrasound_roi(image, mask)
                # print("after crop : ", image.size)
            self.cache[f"{dataset}_{ID}"] = (image, mask, seg_type, from_dataset)

        if dataset == "DDTI" and self.histo_match:
            image = self.histogram_match(image, self.histo_match_image)
            
        
        
        image_tensor, mask_tensor = self.transform(image, mask, self.image_size)

        

        
        mask_tensor = (mask_tensor > 0.5).float()

        contour_image = self.get_contour_from_mask_tensor(mask_tensor)
        
        seg_type = torch.tensor(seg_type)

        # print(image_tensor.shape)
        # print(mask_tensor.shape)
        # print(contour_image.shape)
        if self.return_from_dataset:
            return image_tensor, mask_tensor, contour_image, seg_type, from_dataset
        else:
            return image_tensor, mask_tensor, contour_image, seg_type

    def get_contour_from_mask_tensor(self, mask_tensor):
        """
        Extract contours from a transformed segmentation mask tensor.
        Args:
            mask_tensor (torch.Tensor): Shape (1, H, W) or (H, W)
        Returns:
            contours: List of contour points
            contour_image: Image with contours drawn (for visualization)
        """
        # Remove channel if exists and convert to NumPy
        if mask_tensor.dim() == 3:
            mask_np = mask_tensor.squeeze().cpu().numpy()
        else:
            mask_np = mask_tensor.cpu().numpy()
    
        # Binarize and convert to uint8
        mask_bin = (mask_np > 0.5).astype(np.uint8) * 255
    
        # OpenCV expects a CV_8UC1 image
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Optional: visualize the contour
        contour_image = np.zeros_like(mask_bin)
        cv2.drawContours(contour_image, contours, -1, 255, thickness=3)
        contour_image = contour_image.astype(np.float32) / 255.0
        
        contour_image = torch.tensor(contour_image)
        contour_image = torch.unsqueeze(contour_image, 0)
    
        return contour_image
    def crop_ddti_ultrasound_roi(self, pil_image, pil_mask, crop_ratio=0.8):
    
        # 中心裁切
        w, h = pil_image.size
        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        cropped_image = pil_image.crop((left, top, left + crop_w, top + crop_h))
        cropped_mask = pil_mask.crop((left, top, left + crop_w, top + crop_h))
        
        # 對比增強（幫助分辨紋理）
        enhancer = ImageEnhance.Contrast(cropped_image)
        cropped_image = enhancer.enhance(1.5)  # 可微調，1.5–2 之間效果通常最好

    
        return cropped_image, cropped_mask
    def histogram_match(self, source_pil, reference_pil):
        """
        將 source_pil 的 histogram 調整成與 reference_pil 接近
        """
        source = np.array(source_pil.convert("L"))
        reference = np.array(reference_pil.convert("L"))

    
        matched = match_histograms(source, reference, channel_axis=None)
        matched_img = Image.fromarray(np.uint8(matched))
        return matched_img