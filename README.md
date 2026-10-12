# MRI–CT Brain Image Fusion (Modular Python Project)

This refactors the uploaded Colab notebook into smaller files so the dataset,
model, training, inference, and evaluation code can be changed independently.

> **Scope:** This implementation performs MRI–CT image fusion. It does not
> currently detect, segment, or predict tumors. The discriminator's score is
> not a tumor prediction and must not be used for medical decisions.

## 1. Project structure

```text
mri_ct_fusion_project/
├── config.py        # Settings and paths
├── dataset.py       # Loads and preprocesses paired MRI/CT images
├── models.py        # Generator, encoder, residual/decoder blocks, discriminator
├── train.py         # GAN training loop and checkpoints
├── infer.py         # Generate a fused image from a trained checkpoint
├── evaluate.py      # SSIM, PSNR, MSE, entropy
├── utils.py         # Reproducibility, plotting, image conversion
├── main.py          # Simple end-to-end entry point
├── requirements.txt
├── data/
│   └── final year dataset/
│       ├── mri/
│       └── ct/
├── checkpoints/     # Saved model weights (created automatically)
└── outputs/         # Generated images (created automatically)
```

## 2. Install dependencies

Use Python 3.10 or 3.11 in a virtual environment where possible:

```bash
pip install -r requirements.txt
```

For Google Colab, run this in a notebook cell from the project folder:

```python
!pip install -r requirements.txt
```

## 3. Put the dataset in the expected layout

```text
data/final year dataset/mri/img1.png
data/final year dataset/ct/img1.png
```

The MRI and CT folders should contain matching scans, ideally with matching
filenames. The loader first matches filenames. If names differ but counts match,
it falls back to sorted positional pairing and prints a warning. Verify pairing
yourself; resizing does not register or align scans.

If your ZIP extracts to `/content/final year dataset/`, set the data root to that
folder when running in Colab.

## 4. Train the model

From a terminal, run:

```bash
python train.py --data-root "data/final year dataset" --epochs 100 --batch-size 4
```

For a quick smoke test, use `--epochs 1`. That confirms the training loop runs;
it does **not** mean the model is trained well.

In Colab, after uploading/extracting the dataset, run:

```python
!python train.py --data-root "/content/final year dataset" --epochs 100 --batch-size 4
```

Training can take a long time. Use a GPU runtime if available. If the original
dataset folder has another name, change `--data-root` to the actual path.

## 5. Generate a fused image after training

```bash
python infer.py \
  --mri "data/final year dataset/mri/img1.png" \
  --ct "data/final year dataset/ct/img1.png" \
  --checkpoint "checkpoints/generator_model.pth"
```

Colab example:

```python
!python infer.py --mri "/content/final year dataset/mri/img1.png" --ct "/content/final year dataset/ct/img1.png" --checkpoint "checkpoints/generator_model.pth"
```

Outputs are saved to `outputs/fused_output.png` and
`outputs/comparison_result.png`.

## 6. Evaluate a result

```bash
python evaluate.py \
  --mri "data/final year dataset/mri/img1.png" \
  --ct "data/final year dataset/ct/img1.png" \
  --fused "outputs/fused_output.png"
```

SSIM, PSNR, and MSE compare the fused image with each source image; entropy
summarizes intensity variability. These metrics alone do not prove clinical
quality, and without a ground-truth fused image they are not a definitive
measure of fusion correctness.

## 7. Important implementation notes

- The architecture is based on the uploaded code: two encoders, concatenation,
  fusion convolution, three residual blocks, three PixelShuffle decoder blocks,
  Tanh output, and PatchGAN discriminator.
- The original training idea is retained: CT is treated as the real image for
  discriminator training; the generator is also encouraged to resemble the
  average of MRI and CT with L1 and SSIM losses. This objective may need research
  and validation before claiming high-quality fusion.
- The model does not include attention, image registration, or a tumor analysis
  module. Those should be implemented and validated as separate project stages.
- Checkpoint files can be large; do not commit them to Git.
- The original Colab `files.upload()`, `files.download()`, `!ls`, and `!pip`
  notebook commands were moved out of reusable Python modules.
