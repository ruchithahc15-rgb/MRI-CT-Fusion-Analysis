"""Evaluate a fused image against its MRI and CT inputs."""
import argparse
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from skimage.measure import shannon_entropy
from utils import tensor_to_uint8

def evaluate_arrays(mri, ct, fused):
    results = {}
    for name, source in (("MRI", mri), ("CT", ct)):
        results[f"SSIM {name}-Fused"] = structural_similarity(source, fused, data_range=255)
        results[f"PSNR {name}-Fused"] = peak_signal_noise_ratio(source, fused, data_range=255)
        results[f"MSE {name}-Fused"] = mean_squared_error(source, fused)
    results["Fused entropy"] = shannon_entropy(fused)
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mri", required=True, help="Path to MRI image")
    p.add_argument("--ct", required=True, help="Path to CT image")
    p.add_argument("--fused", required=True, help="Path to generated fused image")
    args = p.parse_args()
    import cv2
    images = []
    for path in (args.mri, args.ct, args.fused):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        images.append(cv2.resize(image, (256, 256)))
    for key, value in evaluate_arrays(*images).items():
        print(f"{key}: {value:.5f}")

if __name__ == "__main__":
    main()
