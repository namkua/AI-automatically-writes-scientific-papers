# ConvNeur Template — AI Scientist

Official implementation of **ConvNeur** (Global-Local Decoupled vision architecture) for automated scientific discovery:

> **"Efficiency Follows Global-Local Decoupling"** — CVPR 2026  
> Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Efficiency_Follows_Global-Local_Decoupling_CVPR_2026_paper.html  
> Official Codebase: https://github.com/NUST-Machine-Intelligence-Laboratory/ConvNeur.git

## Architecture Overview

ConvNeur decouples visual processing into two specialized branches per block:

```
Input Feature Map (B, C, H, W)
        │
  ┌─────┴────────────────┐
  ▼                      ▼
Locality Branch        Neural Memory Branch
(ConvNeXt 7x7 DW       (Titans-style NeuralMemory
 + GlobalNorm + PW)     Long-term token memory)
  │                      │
  │                      ▼
  │               x_mem = sigmoid(proj_out(x_mem))
  └──────────┬───────────┘
             ▼
        x = x_mem * x   (Multiplicative Gated Modulation)
             ▼
     + DropPath + Residual
```

## Supported Datasets

| Dataset | Classes | Resolution | Role in Evaluation |
|---|---|---|---|
| **CIFAR-10** | 10 | 32x32 | Coarse object recognition baseline |
| **CIFAR-100** | 100 | 32x32 | Fine-grained discrimination |
| **Tiny-ImageNet-200** | 200 | 64x64 | Scaled miniature of ImageNet-1K (Academic benchmark) |

## Quick Start & Baseline Execution

```bash
# Activate environment
conda activate ai-scientist

# 1. Run baseline on CIFAR-10
python experiment.py --out_dir run_0 --dataset cifar10

# 2. Run baseline on CIFAR-100
python experiment.py --out_dir run_0 --dataset cifar100

# 3. Run baseline on Tiny-ImageNet-200 (64x64, 200 classes)
python experiment.py --out_dir run_0 --dataset tiny_imagenet

# 4. Generate Publication Figures
python plot.py
```

## Running AI Scientist

```bash
python launch_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment convneur \
    --num-ideas 3
```
