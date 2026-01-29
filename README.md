# PhysicsAgentABM: Physics-Guided Generative Agent-Based Modeling

**ICML 2026 Submission - Code Repository**

This repository contains the implementation of PhysicsAgentABM, a hierarchical neuro-symbolic framework for scalable and calibrated agent-based modeling with LLMs, and ANCHOR, a novel LLM-agent-driven clustering mechanism.

## Overview

PhysicsAgentABM shifts inference from individual agents to adaptive agent clusters, combining:
- **Symbolic reasoning** via state-specialized agents and meta-agents
- **Neural transition models** for temporal and interaction dynamics
- **Uncertainty-aware epistemic fusion** for calibrated predictions
- **ANCHOR clustering** for behaviorally coherent abstraction

We evaluate on three domains:
- **Epidemiology**: Singapore COVID-19 (1,000 agents, 83 days)
- **Finance**: Market sentiment diffusion (100 traders, 184 days)
- **Social Sciences**: Climate attention lifecycle (250 agents, 90 days)

## Setup

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate epidemic-reasoning
```

### 2. API Configuration

Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Data Preparation
```bash
# Download datasets (instructions in data/README.md)
cd data
bash download_datasets.sh
cd ..
```

## Repository Structure
```
.
├── src/
│   ├── clustering/          # ANCHOR implementation
│   ├── agents/              # Meta, State, and Entity agents
│   ├── models/              # Neural pathway (GNN-LSTM)
│   ├── fusion/              # Epistemic fusion module
│   └── utils/               # Data loaders, metrics
├── experiments/
│   ├── epidemiology/        # COVID-19 simulation
│   ├── finance/             # Market sentiment diffusion
│   └── social/              # Attention lifecycle
├── data/                    # Datasets (see data/README.md)
├── configs/                 # Experiment configurations
├── results/                 # Output logs and figures
└── environment.yml          # Conda environment
```

## Running Experiments

### Epidemiology (Singapore COVID-19)
```bash
python experiments/epidemiology/run_simulation.py \
    --config configs/epidemiology.yaml \
    --n_agents 1000 \
    --n_clusters 4 \
    --eval_mode rolling_window
```

### Finance (Market Sentiment)
```bash
python experiments/finance/run_simulation.py \
    --config configs/finance.yaml \
    --n_agents 100 \
    --n_clusters 5 \
    --eval_mode rolling_window
```

### Social Sciences (Attention Lifecycle)
```bash
python experiments/social/run_simulation.py \
    --config configs/social.yaml \
    --n_agents 250 \
    --n_clusters 3 \
    --eval_mode rolling_window
```

### Reproducing All Results
```bash
# Run all experiments with default settings
bash scripts/reproduce_all.sh

# Results will be saved to results/
```

## Key Components

### ANCHOR Clustering
```bash
# Run standalone ANCHOR clustering
python src/clustering/run_anchor.py \
    --dataset epidemiology \
    --n_coarse_clusters 10 \
    --n_final_clusters 4
```

### Ablation Studies
```bash
# Architecture ablation
python experiments/ablation/architecture.py --config configs/ablation.yaml

# ANCHOR ablation
python experiments/ablation/clustering.py --config configs/ablation.yaml
```

## Evaluation Metrics

Our evaluation focuses on event-time accuracy and calibration:

- **EETE** (Expected Event Time Error): Temporal alignment ↓
- **ET-F1** (Event-Type Macro-F1): Transition detection ↑
- **NLL** (Negative Log-Likelihood): Probabilistic fit ↓
- **Brier** (Brier Score): Calibration sharpness ↓

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (tested on A100)
- **RAM**: ≥32GB
- **Storage**: ≥50GB for datasets and results

Experiments use asynchronous API calls (50 concurrent) to manage LLM inference latency.

## Expected Runtime

- **Epidemiology** (N=1000, T=83): ~60 mins
- **Finance** (N=100, T=184): ~35 mins
- **Social** (N=250, T=90): ~40 mins

Times reported for rolling-window evaluation with α=1.0.

## Citation
```bibtex
@inproceedings{physicsagentabm2026,
  title={PhysicsAgentABM: Physics-Guided Generative Agent-Based Modeling},
  author={Anonymous Authors},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Contact

For questions regarding code or experiments, please open an issue or contact: `anon.email@domain.com`

## License

This code is released for review purposes only. License details will be provided upon acceptance.
