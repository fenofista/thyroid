import matplotlib.pyplot as plt
import os

def save_2x2_comparison(nodule_contour_mask, nodule_contour_show, nodule_image_mask, nodule_image_show, save_path="./output_images/test/train/", filename="nodule_plot.png"):
    # Make sure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    # Create the figure
    plt.figure(figsize=(10, 8))

    # Subplot 1
    plt.subplot(2, 2, 1)
    plt.imshow(nodule_contour_mask[0][0].detach().cpu().numpy(), cmap='gray')
    plt.title("Contour Mask")
    plt.axis("off")

    # Subplot 2
    plt.subplot(2, 2, 2)
    plt.imshow(nodule_contour_show[0][0].detach().cpu().numpy(), cmap='gray')
    plt.title("Contour Pred")
    plt.axis("off")

    # Subplot 3
    plt.subplot(2, 2, 3)
    plt.imshow(nodule_image_mask[0][0].detach().cpu().numpy(), cmap='gray')
    plt.title("Image Mask")
    plt.axis("off")

    # Subplot 4
    plt.subplot(2, 2, 4)
    plt.imshow(nodule_image_show[0][0].detach().cpu().numpy(), cmap='gray')
    plt.title("Image Pred")
    plt.axis("off")

    # Save
    full_path = os.path.join(save_path, filename)
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved to {full_path}")


import torch
import numpy as np
import cv2

def postprocess_logits_mask(logits_mask, threshold=0.5):
    """
    logits_mask: torch.Tensor of shape [B, 1, H, W], raw logits
    threshold: value for converting logits to binary mask after sigmoid
    returns: torch.Tensor of shape [B, 1, H, W] with only the largest component retained
    """
    # Step 1: Convert logits to probabilities
    prob_mask = torch.sigmoid(logits_mask)

    # Step 2: Threshold to get binary mask
    binary_mask = (prob_mask > threshold).float()

    # Step 3: Remove all but the largest connected component
    binary_mask = binary_mask.squeeze(1)  # [B, H, W]
    cleaned_batch = []

    for mask in binary_mask:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

        if num_labels <= 1:
            cleaned = np.zeros_like(mask_np)
        else:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_label).astype(np.uint8)

        cleaned_batch.append(torch.from_numpy(cleaned))

    # Stack and return to original shape [B, 1, H, W]
    cleaned_mask = torch.stack(cleaned_batch).unsqueeze(1).float().to(logits_mask.device)

    return cleaned_mask
import torch
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

def postprocess_logits_with_fill(logits_mask, threshold=0.5):
    """
    logits_mask: torch.Tensor of shape [B, 1, H, W], raw logits
    returns: torch.Tensor of shape [B, 1, H, W], cleaned mask
    """
    # Step 1: Apply sigmoid + threshold
    prob_mask = torch.sigmoid(logits_mask)
    binary_mask = (prob_mask > threshold).float()
    binary_mask = binary_mask.squeeze(1)  # [B, H, W]
    
    cleaned_batch = []

    for mask in binary_mask:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Step 2: Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        if num_labels <= 1:
            largest_component = np.zeros_like(mask_np)
        else:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_component = (labels == largest_label).astype(np.uint8)
        
        # Step 3: Fill holes in the largest component
        filled = binary_fill_holes(largest_component).astype(np.uint8)

        cleaned_batch.append(torch.from_numpy(filled))

    return torch.stack(cleaned_batch).unsqueeze(1).float().to(logits_mask.device)