# Simple U-Net Satellite Segmentation

Compact PyTorch implementation of a U-Net-style semantic segmentation workflow for satellite land-cover images.

This repository is a fork of the Preligens ENS Challenge Data benchmark. The `simple_u-net` version keeps a smaller, easier-to-read training script and utility package for portfolio and experimentation purposes.

## What It Shows

- U-Net-style segmentation model in PyTorch
- GeoTIFF image and mask loading utilities
- train/validation split and dataloaders
- weighted cross-entropy for imbalanced classes
- gradient clipping to stabilize training
- validation loss and mean IoU tracking
- prediction visualization helpers

## Structure

```text
.
├── main.py              # Training, validation and visualization entry point
├── ml_utils/
│   ├── data.py          # Dataset loading and preprocessing
│   ├── model.py         # Simple U-Net model and losses
│   └── viz.py           # Visualization helpers
├── pyproject.toml
└── uv.lock
```

## Dataset

The dataset is not included in this repository.

Expected local structure:

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

## Notes

- This is an experimentation branch, not a packaged production library.
- Large datasets, model checkpoints and generated artifacts are ignored by Git.
- The original benchmark repository remains credited through the fork history and license.
