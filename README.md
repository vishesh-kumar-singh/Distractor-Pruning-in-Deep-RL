# EMA-Tree SAC: Dynamic Distractor Pruning in Deep RL

This repository contains the official implementation of **EMA-Tree SAC**, a computationally efficient algorithm that dynamically prunes high-variance noise and spurious correlations from continuous control state spaces.

## Repository Structure

- `agents/`: Contains the baseline Soft Actor-Critic (`sac.py`) and our novel agent (`ema_tree_sac.py`).
- `envs/`: Contains the `Distracting HalfCheetah` benchmark environment.
- `scripts/train_cheetah.py`: The main training loop.
- `scripts/plot_cheetah.py`: Generates the side-by-side performance and feature pruning plots.
- `run_experiments.sh`: An automated, resumable bash script to run the 10-seed experiment sweep.
- `Paper.tex` & `tmlr.bib`: The LaTeX source code for the manuscript.

## Getting Started

1. Set up the environment: `conda activate gymnasium`
2. Run the experiments: `bash run_experiments.sh`
3. The high-resolution results plot will be saved to `plots/combined_results.pdf`.
