"""Shared helpers for reproducibility, plots, and image saving."""
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tensor_to_uint8(tensor):
    """Convert a normalized [-1, 1] grayscale tensor to uint8."""
    arr = tensor.detach().cpu().squeeze().numpy()
    arr = np.clip((arr + 1.0) * 127.5, 0, 255)
    return arr.astype(np.uint8)

def save_comparison(mri, ct, fused, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mri_img, ct_img, fused_img = map(tensor_to_uint8, (mri, ct, fused))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, (mri_img, ct_img, fused_img),
                              ("MRI Image", "CT Image", "Fused Image")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return mri_img, ct_img, fused_img

def save_gray_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
