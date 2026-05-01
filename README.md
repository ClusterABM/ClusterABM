# ClusterABM: Scalable Agent-Based Modeling under Partial Observability via Cluster-Level Inference


This repository contains the implementation of ClusterABM, a hierarchical neuro-symbolic framework for scalable and calibrated agent-based modeling with LLMs, and ANCHOR, a novel LLM-agent-driven clustering mechanism.

## Overview

ClusterABM shifts inference from individual agents to adaptive agent clusters, combining:
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

### 3. Data Setup

Place the Singapore COVID-19 dataset in `data/raw/`:
- `singapore_covid19_cases.csv` (MOH case records)

See `data/README.md` for data sources and preprocessing details.

## Repository Structure
```
.
├── src/
│   ├── clustering/          # ANCHOR implementation
│   ├── agents/              # Meta, State, and Entity agents
│   ├── neural/              # Neural pathway (multimodal)
│   ├── knowledge/           # Knowledge database for Agentic RAG
│   └── simulation/          # GABM simulation setup
├── config/                  # Configuration files
├── data_processing/         # Data preprocessing utilities
├── data/
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Processed agent data
│   └── results/             # Simulation outputs
├── scripts/                 # Execution scripts (see below)
├── environment.yml          # Conda environment
└── .env                     # API keys (create this)
```

## Running Experiments

### Complete Pipeline (Singapore COVID-19)

Execute the following scripts in order:

#### 1. Extract Agent Data
```bash
python scripts/extract_agent_data.py data/raw/singapore_covid19_cases.csv
```
Processes MOH case records into agent profiles with demographics, contact networks, and infection timelines.

#### 2. Train Graph Embeddings
```bash
python scripts/train_graphsage.py
```
Trains GraphSAGE encoder on contact network structure and agent attributes.

#### 3. Create Knowledge Base
```bash
python scripts/create_knowledge_base.py
```
Builds domain knowledge database for symbolic agents (policies, epidemiological rules, context).

#### 4. Collect Behavioral Traces
```bash
python scripts/collect_reasoning_traces.py
```
Runs mini-simulations to collect agent reasoning traces for motif discovery.

#### 5. ANCHOR Clustering
```bash
python scripts/cluster_agents_hsbc.py

# Custom parameters
python scripts/cluster_agents_hsbc.py --k-fine 8 --k-motifs 20 --alpha 0.2 --gamma 0.6

# Ablations
python scripts/cluster_agents_hsbc.py --no-motifs        # Remove behavioral motifs
python scripts/cluster_agents_hsbc.py --no-contrastive   # Remove anchor-guided learning
python scripts/cluster_agents_hsbc.py --no-graph         # Remove graph structure
```
Executes ANCHOR clustering pipeline: spectral → motif discovery → anchor-guided refinement.

#### 6. Train Neural Pathway
```bash
python scripts/train_neural_pathway.py
```
Trains multimodal neural transition model on historical data.

#### 7. Run Simulation
```bash
python scripts/run_rolling_window_simulation.py
```
Executes rolling-window forecasting with epistemic fusion (28-day lookback, 7-day horizon).

#### 8. Analyze Results
```bash
python scripts/analyze_results.py
```
Computes evaluation metrics (EETE, ET-F1, NLL, Brier) and generates performance tables.

#### 9. Visualize Results
```bash
python scripts/visualize_epidemic.py
```
Generates trajectory plots, calibration curves, and cluster analyses.

### Quick Reproduction
```bash
# Run complete pipeline
bash scripts/run_all.sh

# Results will be saved to data/results/
```

## Key Components

### ANCHOR Clustering Options

The clustering script supports multiple configurations:
```bash
# Specify data source
python scripts/cluster_agents_hsbc.py --data-source singapore

# Adjust cluster granularity
python scripts/cluster_agents_hsbc.py --k-fine 8  # Fine clusters (default: 4)

# Modify motif discovery
python scripts/cluster_agents_hsbc.py --k-motifs 20  # Behavioral motifs (default: 15)

# Tune fusion weights
python scripts/cluster_agents_hsbc.py --alpha 0.2 --gamma 0.6  # Network vs behavior
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
- **Storage**: ≥20GB for datasets and results

Experiments use asynchronous API calls (50 concurrent) to manage LLM inference latency.

## Expected Runtime

**Singapore COVID-19 (N=1000, T=83)**:
- Data extraction: ~2 mins
- GraphSAGE training: ~5 mins
- Knowledge base creation: ~3 mins
- Reasoning traces: ~15 mins
- ANCHOR clustering: ~10 mins
- Neural training: ~8 mins
- Simulation: ~60 mins
- **Total**: ~103 mins

Times reported for rolling-window evaluation with α=1.0 on A100.

## Configuration

Edit `config/*.yaml` files to modify:
- Agent population size
- Cluster counts
- Neural architecture
- Simulation parameters
- Evaluation protocols

## Output Structure

Results are organized in `data/results/`:
```
data/results/
├── clusters/              # ANCHOR outputs
│   ├── cluster_assignments.json
│   ├── motif_profiles.pkl
│   └── anchor_agents.json
├── trajectories/          # Simulation outputs
│   ├── seird_timeseries.csv
│   └── transition_probabilities.pkl
├── metrics/               # Evaluation results
│   ├── rolling_window_metrics.csv
│   └── calibration_data.pkl
└── figures/               # Visualizations
    ├── epidemic_trajectory.png
    ├── cluster_analysis.png
    └── calibration_plots.png
```

## License

This code is released for review purposes only. License details will be provided upon acceptance. 
