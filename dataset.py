"""Dataset loading and preprocessing for paired MRI/CT images."""
from pathlib import Path
import warnings
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def build_transform(image_size: int = 256):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

class FusionDataset(Dataset):
    """Loads grayscale MRI/CT pairs.

    It matches equal filenames first. If names differ, it falls back to sorted
    positional pairing only when both folders contain the same number of images.
    For valid fusion, each pair must depict corresponding anatomy.
    """
    def __init__(self, mri_dir, ct_dir, image_size=256, allow_positional_pairing=True):
        self.mri_dir = Path(mri_dir)
        self.ct_dir = Path(ct_dir)
        if not self.mri_dir.is_dir():
            raise FileNotFoundError(f"MRI folder not found: {self.mri_dir}")
        if not self.ct_dir.is_dir():
            raise FileNotFoundError(f"CT folder not found: {self.ct_dir}")

        mri_files = sorted(p for p in self.mri_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        ct_files = sorted(p for p in self.ct_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not mri_files or not ct_files:
            raise ValueError("No supported images found in the MRI and/or CT folder.")

        mri_by_name = {p.name: p for p in mri_files}
        ct_by_name = {p.name: p for p in ct_files}
        common_names = sorted(set(mri_by_name) & set(ct_by_name))
        if len(common_names) == len(mri_files) == len(ct_files):
            self.pairs = [(mri_by_name[n], ct_by_name[n]) for n in common_names]
            print(f"Matched {len(self.pairs)} MRI/CT pairs by filename.")
        elif allow_positional_pairing and len(mri_files) == len(ct_files):
            warnings.warn(
                "MRI and CT filenames do not match exactly. Pairing files by sorted "
                "position. Verify that each pair belongs to the same subject and scan.",
                stacklevel=2,
            )
            self.pairs = list(zip(mri_files, ct_files))
        else:
            raise ValueError(
                f"Could not pair images safely: {len(mri_files)} MRI and "
                f"{len(ct_files)} CT files. Ensure matching filenames or equal counts."
            )
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mri_path, ct_path = self.pairs[idx]
        mri = cv2.imread(str(mri_path), cv2.IMREAD_GRAYSCALE)
        ct = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
        if mri is None:
            raise ValueError(f"Could not read MRI image: {mri_path}")
        if ct is None:
            raise ValueError(f"Could not read CT image: {ct_path}")
        return self.transform(mri), self.transform(ct)
