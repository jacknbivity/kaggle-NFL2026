# NFL Big Data Bowl 2026 - Prediction

> Kaggle Competition: [NFL Big Data Bowl 2026 - Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

## 🏆 Result

**21st out of 1899 teams** (Top ~1.1%)

![Ranking](./RANK.png)

## 👤 My Contribution

This repository contains only the modeling work I ([jacknbivity](https://github.com/jacknbivity)) was responsible for, covering two deep learning trajectory prediction approaches. Other teammates' work — including data cleaning, feature engineering pipeline integration, and post-processing optimization — is not detailed here.

My core contributions:
- 🧠 Designed and implemented **ST-GRU** and **ST-Transformer** spatiotemporal sequence prediction models
- 🎯 Proposed **Temporal Huber Loss** — a time-weighted loss function with temporal decay penalty for long-horizon predictions
- 🔀 **Direction unification** + **relative coordinate transformation** preprocessing pipeline, eliminating play-direction heterogeneity
- 📐 **Frame Spatial Attention** + **Relative Position Bias** for intra-frame player interaction modeling
- 🔢 **Fourier Feature Encoder** / **RBF Encoder** for high-frequency positional encoding
- ⚡ Full-pipeline **memory optimization** (float32 quantization, Polars-based efficient data processing, multi-process parallel loading)
- 📊 **GroupKFold cross-validation** + multi-seed ensemble

---

This project provides two deep learning models for predicting NFL player trajectories:

| Model | File | Description |
|------|------|-------------|
| **ST-GRU** | `519-ST-GRU.py` | SpatioTemporal GRU-based sequence prediction with spatial attention and Fourier feature encoding |
| **ST-Transformer** | `519-STTransformer.py` | SpatioTemporal Transformer-based sequence prediction with stronger spatiotemporal modeling |

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA recommended)
- See [requirements.txt](requirements.txt)

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

Place the Kaggle competition data under `./nfl-big-data-bowl-2026-prediction/` with the following structure:

```
nfl-big-data-bowl-2026-prediction/
├── train/
│   ├── input_2023_w01.csv
│   ├── output_2023_w01.csv
│   ├── ...
│   ├── input_2023_w09.csv
│   └── output_2023_w09.csv
└── test/
    └── ...
```

### Training

```bash
# Train ST-GRU model
python 519-ST-GRU.py --mode train

# Train ST-Transformer model
python 519-STTransformer.py --mode train
```

### Inference

```bash
# Run ST-GRU inference
python 519-ST-GRU.py --mode infer

# Run ST-Transformer inference
python 519-STTransformer.py --mode infer
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode: `train` or `infer` | `infer` |
| `--model_path` | Path to save/load models | `./saved_models_rnn` |
| `--use_cache` | Use cached sequences if available | `True` |
| `--no_cache` | Force rebuild sequences (ignore cache) | — |

## Methodology

### Feature Engineering

- **Direction unification**: Mirror all rightward plays to left orientation for consistent learning
- **Coordinate transformation**: Convert absolute coordinates to ball-landing-relative coordinates
- **Advanced features**: Velocity/acceleration components, momentum, kinetic energy, spatial field position features
- **Spatiotemporal sequences**: Construct plays as `[B, T, N, F]` format spatiotemporal sequences

### Model Architecture

#### ST-GRU (SpatioTemporal GRU)
Design pipeline: **Intra-frame Spatial Attention → Inter-frame GRU Temporal Modeling → Attention Pooling → Segmented Prediction**

| Module | Description |
|--------|-------------|
| `FourierFeatureEncoder` | Multi-frequency Fourier encoding (bands=1,2,4,8,16), mapping low-dim coordinates to high-dim sine/cosine space |
| `RBFEncoder` | Radial Basis Function encoding for non-linear distance feature expansion |
| `RelativePositionBias` | Relative position bias enhancing spatial attention's geometric awareness between players |
| `FrameSpatialAttention` | Intra-frame multi-head spatial attention modeling all-player interactions at each timestep |
| `SpatioTemporal_GRU` | Stacked GRU for inter-frame temporal encoding with MultiheadAttention pooling |
| `ResidualMLP` | Residual MLP prediction head with segmented short/mid/long horizon outputs |
| `TemporalHuber` | Time-weighted Huber loss: temporal decay + L2 smoothness regularization |

#### ST-Transformer (SpatioTemporal Transformer)
Design pipeline: **Input Embedding + Temporal Positional Encoding → Spatial Transformer (intra-frame) → Temporal Transformer (inter-frame) → Linear Prediction Head**

| Module | Description |
|--------|-------------|
| `embedding` | Feature embedding layer mapping input to hidden_dim |
| `temporal_pos_embedding` | Learnable temporal positional encoding |
| `spatial_transformer` | nn.TransformerEncoder with GELU activation, modeling intra-frame player spatial interactions |
| `temporal_transformer` | nn.TransformerEncoder modeling temporal dependencies |
| `head` | Linear layer outputting horizon×2 dimensional displacement predictions |

### Loss Function: Temporal Huber Loss

$$
L = \underbrace{\frac{\sum_{t} w_t \cdot \text{Huber}(y_t, \hat{y}_t)}{\sum w_t}}_{\text{Time-Decayed Huber}} + \lambda \cdot \underbrace{\frac{\sum_{t} \|\Delta^2 \hat{y}_t\|^2}{\sum m_t}}_{\text{2nd-Order Smoothness Regularization}}
$$

Where $w_t = e^{-\alpha t}$ is the temporal decay weight — distant frames receive smaller penalties (encouraging the model to prioritize near-term accuracy); $\lambda$ controls trajectory smoothness.

### Training Strategy

- **Cross-Validation**: GroupKFold (grouped by game_id), ST-GRU 5-fold / ST-Transformer 10-fold
- **Early Stopping**: patience=30 epochs
- **Optimizer**: Adam, learning rate 1e-3
- **Batch**: Batch Size 128, window size 8, hidden dimension 128
- **Regularization**: Dropout 0.1, LayerNorm, multi-seed ensemble
- **Data Quality**: Truncate anomalous plays exceeding 50 frames (5 seconds)

## Project Structure

```
.
├── 519-ST-GRU.py          # ST-GRU model (primary)
├── 519-STTransformer.py   # ST-Transformer model
├── RANK.png              # Competition ranking screenshot
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file (Chinese)
├── README_EN.md          # This file (English)
├── log/                  # Run logs (auto-generated)
│   └── YYYYMMDD_HHMMSS/  # Timestamped log directories
├── saved_models_rnn/     # Model save directory
└── nfl-big-data-bowl-2026-prediction/  # Competition data (download separately)
```

## Acknowledgments

- [NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) organizers
- Kaggle Community

## License

MIT License
