"""Train the GAN. Run: python train.py --data-root "data/final year dataset" """
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from config import Config
from dataset import FusionDataset
from models import Generator, Discriminator
from utils import set_seed

def train(config: Config):
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = FusionDataset(config.mri_dir, config.ct_dir, config.image_size)
    if len(dataset) < 1:
        raise ValueError("Dataset is empty.")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, drop_last=False)

    device = config.device
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate,
                                   betas=(config.beta1, config.beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate,
                                   betas=(config.beta1, config.beta2))

    print(f"Device: {device} | image pairs: {len(dataset)} | epochs: {config.epochs}")
    for epoch in range(config.epochs):
        generator.train()
        discriminator.train()
        total_g, total_d, batches = 0.0, 0.0, 0

        for mri, ct in loader:
            mri, ct = mri.to(device), ct.to(device)
            fused = generator(mri, ct)

            # Train discriminator: original implementation treats CT as "real".
            # This is retained to match the uploaded project's training design.
            real_logits = discriminator(ct)
            fake_logits = discriminator(fused.detach())
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            d_loss = (adversarial_loss(real_logits, real_labels) +
                      adversarial_loss(fake_logits, fake_labels)) / 2
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            # Train generator to fool discriminator and resemble the input average.
            fake_logits_for_g = discriminator(fused)
            target = (mri + ct) / 2
            g_gan = adversarial_loss(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            g_l1 = l1_loss(fused, target)
            g_ssim = 1.0 - ssim(fused, target, data_range=2.0, size_average=True)
            g_loss = g_gan + config.lambda_l1 * g_l1 + config.lambda_ssim * g_ssim

            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()

            total_g += g_loss.item()
            total_d += d_loss.item()
            batches += 1

        print(f"Epoch [{epoch + 1}/{config.epochs}] "
              f"D Loss: {total_d / max(batches, 1):.4f} "
              f"G Loss: {total_g / max(batches, 1):.4f}")

        # Save a resumable checkpoint after every epoch.
        torch.save({
            "epoch": epoch + 1,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "config": vars(config),
        }, config.checkpoint_dir / "latest_checkpoint.pth")

    torch.save(generator.state_dict(), config.checkpoint_dir / "generator_model.pth")
    torch.save(discriminator.state_dict(), config.checkpoint_dir / "discriminator_model.pth")
    print(f"Training complete. Models saved in: {config.checkpoint_dir.resolve()}")
    return generator, discriminator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/final year dataset",
                   help="Folder containing mri/ and ct/ subfolders")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = Config(data_root=Path(args.data_root), epochs=args.epochs,
                 batch_size=args.batch_size, image_size=args.image_size,
                 output_dir=Path(args.output_dir),
                 checkpoint_dir=Path(args.checkpoint_dir))
    train(cfg)
