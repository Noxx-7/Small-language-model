# SLM: 1.3B Parameter Language Model Training Pipeline

**Optimized for NVIDIA H200 GPU with HBM3 Memory**

A complete training pipeline for building a 1.3 billion parameter GPT-style language model from scratch, with multi-phase curriculum training and safety alignment.

---

## Features

- **1.3B Parameter Transformer** — Custom GPT architecture with modern optimizations
- **H200/Hopper Optimized** — BF16 precision, Flash Attention, torch.compile support
- **Multi-Phase Training** — Pretraining → SFT → Safety Alignment → Domain Fine-tuning
- **Auto GPU Detection** — Adapts to T4, L4, L40S, A100, H100, H200 automatically
- **Checkpoint Resume** — Resume training from any checkpoint seamlessly

---

## Architecture

| Component | Implementation |
|-----------|---------------|
| Parameters | 1.3B |
| Layers | 24 |
| Hidden Dim | 2048 |
| Attention Heads | 16 |
| Context Length | 2048 tokens |
| Positional Encoding | RoPE (Rotary Position Embedding) |
| Normalization | RMSNorm (Pre-Norm) |
| Activation | SwiGLU |
| Attention | Flash Attention / SDPA |
| Precision | BF16 (Hopper) / FP16 (Ampere) |

---

## Training Phases

### Phase B: Pretraining
- **Dataset:** FineWeb-Edu (high-quality web text)
- **Objective:** Next-token prediction
- **Duration:** ~100K steps

### Phase C: Supervised Fine-Tuning + Safety
- **Datasets:** SlimOrca, WikiQA, TruthfulQA
- **Objective:** Instruction following + factual accuracy
- **Safety:** Refusal training for harmful prompts

### Phase D: Domain Fine-Tuning
- **Custom datasets** for specific use cases
- **LoRA-compatible** for efficient adaptation

---

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers accelerate einops matplotlib numpy tqdm tiktoken
pip install flash-attn --no-build-isolation  # Optional, falls back to SDPA
```

---

## GPU Memory Requirements

| GPU | VRAM | Batch Size | Seq Length | Status |
|-----|------|------------|------------|--------|
| T4 | 16GB | 1 | 1024 | Tight fit |
| L4 | 24GB | 2 | 1024 | Works |
| L40S | 48GB | 4 | 2048 | Recommended |
| A100 | 80GB | 8 | 2048 | Optimal |
| H200 | 141GB | 16 | 2048 | Maximum performance |

---

## Quick Start

```python
# 1. Run the notebook cells in order
# 2. GPU is auto-detected and settings applied
# 3. Training starts automatically with optimal config

# Or load a trained checkpoint:
model.load_state_dict(torch.load('checkpoints/gpt_vibe_1.3B_final.pt'))
```

---

## Key Optimizations

- **Gradient Checkpointing** — Reduces memory by recomputing activations
- **Mixed Precision (BF16)** — 2x memory savings, faster compute
- **Flash Attention** — O(N) memory instead of O(N²)
- **Fused Optimizers** — Faster AdamW with fused kernels
- **Dynamic Batching** — Maximizes GPU utilization

---

## File Structure

```
├── h200-ultimate-model-gpt-training-complete.ipynb  # Main training notebook
├── checkpoints/                                      # Saved model weights
├── logs/                                            # Training logs
└── configs/                                         # Hyperparameter configs
```

---

## Training Metrics

The training loop tracks:
- Cross-entropy loss
- Perplexity
- Tokens per second
- GPU memory usage
- Gradient norms

---

## Lessons Learned

This codebase evolved through 200+ iterations. Key bugs fixed:
- Causal mask dimension ordering
- RoPE frequency computation
- FP16 overflow in attention logits
- Gradient explosion at scale

See the companion book *"200 Iterations: Cracking the Code of the Small Language Model"* for detailed debugging case studies.

---

