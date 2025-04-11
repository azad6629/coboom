# CoBooM: Codebook Guided Bootstrapping for Medical Image Representation Learning

Official implementation of the paper "CoBooM: Codebook Guided Bootstrapping for Medical Image Representation Learning" (MICCAI 2024).

## Overview

CoBooM is a novel self-supervised learning approach for medical image representation learning. This method leverages codebook-guided bootstrapping to improve feature representations for downstream medical imaging tasks.


## Repository Structure

```
├── config/          # Configuration files
├── eval/            # Evaluation scripts
│   ├── eval_nih_fi.ipynb  # Semi-supervised fine-tuning evaluation
│   └── eval_nih_fr.ipynb  # Linear probing evaluation
├── optimizer/       # Optimizer implementations
├── utils/           # Utility functions
├── main.py          # Main script for training
├── pretrain.sh      # Script for pre-training
└── trainer.py       # Training loop implementation
```

## Setup


### Dataset Preparation

This implementation uses the NIH Chest X-ray 14 dataset with the official train and test split.

1. Download the dataset from: https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789610
2. Either:
   - Place the dataset at `/workspace/DATASETS/XRAY_datasets/` (default path)
   - Or update the `DATA_BASE_DIR` in the data directory's constant file to point to your dataset location

## Usage

### Pre-training

To pre-train the model with default settings on the NIH Chest X-ray 14 dataset:

```bash
bash pretrain.sh
```

This will run the model with the following configuration:
- Architecture: ResNet-18
- Dataset: NIH14 (NIH Chest X-ray 14)
- Batch size: 64
- Head dimension: 4096
- Output dimension: 256
- Learning rate: 0.08
- Momentum update coefficient: 0.996

To customize the pre-training, you can modify the command-line arguments in `pretrain.sh` or run the main script directly:

```bash
python main.py -arch resnet18 -dataset NIH14 -gpu 0 -bs 64 -hd 4096 -od 256 -lr 0.08 -mu 0.996 -ver v1
```

### Evaluation

Two evaluation methods are provided:

1. **Semi-supervised Fine-tuning**: Use `eval/eval_nih_fi.ipynb` Jupyter notebook
2. **Linear Probing**: Use `eval/eval_nih_fr.ipynb` Jupyter notebook

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{singh2024coboom,
  title={CoBooM: Codebook Guided Bootstrapping for Medical Image Representation Learning},
  author={Singh, Azad and Mishra, Deepak},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={23--33},
  year={2024},
  organization={Springer}
}
```
