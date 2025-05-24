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