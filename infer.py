"""Generate a fused image using a trained Generator checkpoint."""
import argparse
from pathlib import Path
import cv2
import torch
from dataset import build_transform
from models import Generator
from utils import tensor_to_uint8, save_comparison, save_gray_image

def load_image(path, transform):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return transform(image).unsqueeze(0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mri", required=True, help="Path to MRI image")
    p.add_argument("--ct", required=True, help="Path to matching CT image")
    p.add_argument("--checkpoint", default="checkpoints/generator_model.pth")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--image-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Supports a plain state_dict and a training checkpoint dictionary.
    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        model.load_state_dict(checkpoint["generator"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    transform = build_transform(args.image_size)
    mri = load_image(args.mri, transform).to(device)
    ct = load_image(args.ct, transform).to(device)
    with torch.no_grad():
        fused = model(mri, ct)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fused_img = tensor_to_uint8(fused[0])
    save_gray_image(fused_img, out_dir / "fused_output.png")
    save_comparison(mri[0], ct[0], fused[0], out_dir / "comparison_result.png")
    print(f"Saved: {out_dir / 'fused_output.png'}")
    print(f"Saved: {out_dir / 'comparison_result.png'}")

if __name__ == "__main__":
    main()
