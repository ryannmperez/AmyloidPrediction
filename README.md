# Amyloid Prediction

A deep learning framework for predicting amyloid-forming peptides using transformer-based models and autoencoders.

## Overview

This project uses ProteinBERT and semi-supervised learning to predict whether short peptide sequences (6-mers) have amyloid-forming propensity. The model is trained on the WALTZ-DB dataset and can generate predictions across a 64M peptide manifold.

## Project Structure

```
AmyloidPrediction/
├── Anaconda/              # Conda environment files
├── Datasets/              # Training and testing datasets
├── DeepLearning/
│   ├── Autoencoders/      # Autoencoder models for manifold learning
│   └── Transformers/      # ProteinBERT fine-tuning scripts
└── ProspectivePredictions/  # Predictions on peptide manifold
```

## Installation

### Option 1: Conda (Recommended for GPU support)
```bash
conda env create -f Anaconda/DL.yml
conda activate rapids
```

### Option 2: Pip
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Transformer Model
```bash
cd DeepLearning/Transformers
python HT_CV_2.py
```

### 2. Generate Semi-Supervised Predictions
```bash
python semi_supervised_learning_transformer.py <base_model_path>
```

### 3. Make Prospective Predictions
```bash
python prospective_predictions.py <model_path>
```

## Models

### Transformer (ProteinBERT)
- Base model: [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert)
- Fine-tuned using 5-fold stratified cross-validation
- Semi-supervised learning on pseudo-labeled 64M peptide corpus

### Autoencoders
- Sequence-based autoencoder (one-hot encoding)
- PYSAR feature-based autoencoder
- Used for manifold learning and clustering

## Datasets

- **Training/Testing**: Curated from WALTZ-DB
- **Format**: CSV with columns `Sequence` (6-mer) and `label` (0=non-amyloid, 1=amyloid)

## Scripts

| Script | Description |
|--------|-------------|
| `HT_CV_2.py` | 5-fold cross-validation hyperparameter tuning |
| `semi_supervised_learning_transformer.py` | Generate pseudo-labels and train |
| `prospective_predictions.py` | Predict on 64M peptide manifold |
| `extract_model_embeddings.py` | Extract model embeddings |
| `visualize_model_embeddings.py` | UMAP visualization of embeddings |
| `generate_roc_and_hist_plot.py` | Generate ROC curves and histograms |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{amyloid_prediction,
  author = {Perez, Ryan},
  title = {Amyloid Prediction},
  url = {https://github.com/ryannmperez/AmyloidPrediction}
}
```

## License

See LICENSE file for details.
