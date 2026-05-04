# Llama from Scratch

A step-by-step implementation of a LLaMA-style language model built from the ground up in PyTorch, trained on the Arabic `SaudiIrony` tweet dataset.

## Overview

This notebook walks through building a modern transformer-based language model incrementally, starting from a simple embedding + MLP baseline and progressively adding each architectural innovation from the LLaMA paper — RMSNorm, Rotary Positional Embeddings (RoPE), masked multi-head attention, and SwiGLU activations — until a full multi-layer Llama model is assembled.

## Dataset

- **Source:** [`arbml/SaudiIrony`](https://huggingface.co/datasets/arbml/SaudiIrony) via HuggingFace Datasets
- **Content:** Arabic tweets used for irony detection
- **Tokenization:** Word-level tokenizer built from the dataset vocabulary (no subword splitting)

## Architecture Progression

The notebook builds up the model in stages:

| Stage | Model | Key Feature |
|---|---|---|
| 1 | `SimpleBrokenModel` | Embedding + MLP baseline (buggy softmax placement) |
| 2 | `SimpleModel` | Fixed baseline (cross-entropy loss correctly applied to logits) |
| 3 | `SimpleModel_RMS` | + RMSNorm pre-normalization |
| 4 | `RopeModel` | + RoPE attention (unmasked multi-head) |
| 5 | `RopeModel` (v2) | + Causal masking (autoregressive) |
| 6 | `RopeModel` (v3) | + SwiGLU activation in feed-forward |
| 7 | `Llama` | Full model: stacked `LlamaBlock` layers × 4 |

## Key Components

### RMSNorm
Root Mean Square Layer Normalization — a simpler, faster alternative to LayerNorm used in LLaMA. Applied as pre-normalization before each sub-layer.

### Rotary Positional Embeddings (RoPE)
Position information is injected by rotating query and key vectors using position-dependent rotation matrices. This allows the model to capture relative position between tokens via the inner product `q_m · k_n = f(m - n)`.

### Masked Multi-Head Attention (`RoPEMaskedMultiheadAttention`)
Causal (autoregressive) attention using `is_causal=True` in `F.scaled_dot_product_attention`, ensuring each token can only attend to past tokens.

### SwiGLU
Swish-Gated Linear Unit activation ([Noam Shazeer, 2020](https://arxiv.org/pdf/2002.05202v1.pdf)) used in the feed-forward network, replacing ReLU for smoother gradients.

### LlamaBlock
A single transformer block combining:
- RMSNorm → RoPE Masked Multi-Head Attention → residual connection
- RMSNorm → SwiGLU feed-forward network → residual connection

## Model Configuration

```python
MASTER_CONFIG = {
    "vocab_size": <derived from dataset>,
    "batch_size": 32,
    "context_window": 16,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 4,
    "epochs": 10000,
    "log_interval": 10,
}
```

## Training

- **Optimizer:** Adam (`betas=(0.9, 0.95)`, `weight_decay=0.1`, `lr=1e-3`)
- **Scheduler:** Cosine Annealing LR (`CosineAnnealingLR`, `eta_min=1e-5`)
- **Loss:** Cross-entropy on next-token prediction
- **Split:** 80% train / 10% val / 10% test

## Requirements

```
torch
numpy
matplotlib
pandas
datasets
```

Install with:

```bash
pip install torch numpy matplotlib pandas datasets
```

## Usage

Open and run `Llama_from_scratch.ipynb` end-to-end. Each section builds on the previous, so cells should be run in order.

To generate text after training:

```python
print(generate(llama, MASTER_CONFIG, 500)[0])
```

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/pdf/2002.05202v1.pdf)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [SaudiIrony Dataset](https://huggingface.co/datasets/arbml/SaudiIrony)
