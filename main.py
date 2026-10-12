"""Quick end-to-end workflow: train, then save a sample result."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import FusionDataset
from models import Generator
from train import train
from utils import tensor_to_uint8, save_comparison, save_gray_image

def main():
    config = Config()
    generator, _ = train(config)
    dataset = FusionDataset(config.mri_dir, config.ct_dir, config.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mri, ct = next(iter(loader))
    generator.eval()
    with torch.no_grad():
        fused = generator(mri.to(config.device), ct.to(config.device)).cpu()[0]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_gray_image(tensor_to_uint8(fused), config.output_dir / "fused_output.png")
    save_comparison(mri[0], ct[0], fused, config.output_dir / "comparison_result.png")
    print("Done. Check the outputs/ folder.")

if __name__ == "__main__":
    main()
