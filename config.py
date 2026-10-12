"""Central configuration for MRI–CT image fusion."""
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # Expected dataset layout:
    # data/final year dataset/mri/*.png
    # data/final year dataset/ct/*.png
    data_root: Path = Path("data/final year dataset")
    mri_dir_name: str = "mri"
    ct_dir_name: str = "ct"
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    num_workers: int = 0
    lambda_l1: float = 100.0
    lambda_ssim: float = 10.0
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    seed: int = 42

    @property
    def mri_dir(self) -> Path:
        return self.data_root / self.mri_dir_name

    @property
    def ct_dir(self) -> Path:
        return self.data_root / self.ct_dir_name

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
