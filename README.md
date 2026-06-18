# EMA-Tree SAC: Dynamic Distractor Pruning in Deep RL

This repository contains the official implementation of **EMA-Tree SAC**, a computationally efficient algorithm that dynamically prunes high-variance noise and spurious correlations from continuous control state spaces.

## Repository Structure

- `agents/`: Contains the baseline Soft Actor-Critic (`sac.py`), our novel agent (`ema_tree_sac.py`), and regularized baselines (`l1_sac.py`, `group_lasso_sac.py`).
- `envs/`: Contains the Distracting continuous control environments (`cheetah`, `hopper`, `walker2d`).
- `scripts/train.py`: The main resumable training loop supporting automatic checkpointing.
- `scripts/plot_results.py`: Generates side-by-side performance and feature pruning plots.
- `run_experiments.sh`: An automated, resumable bash script to run the multi-environment, multi-algorithm experiment sweep.
- `Paper.tex` & `tmlr.bib`: The LaTeX source code for the manuscript.
- `requirements.txt`: Project dependencies.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt` (or activate your gymnasium conda environment)
2. Run the experiments: `bash run_experiments.sh`
3. Results and plots will be dynamically saved to `results/$TIMESTEPS/` and `plots/$TIMESTEPS/`.
4. If the script is interrupted, simply rerun it. It will automatically detect checkpoints and resume from the exact timestep it left off!

## Experimental Results

Evaluated over 10 independent seeds across a strict 100,000 timestep horizon, our method demonstrates significant robustness against catastrophic interference:
- **Distracting HalfCheetah:** EMA-Tree SAC yields a **51% improvement** in final evaluation average compared to the standard SAC baseline. Standard $L_1$ regularization catastrophically fails here by blinding the agent to true kinematic correlations.
- **Distracting Hopper:** EMA-Tree SAC yields a **39% improvement** over standard SAC, establishing a stable mask that shields the agent from late-stage value collapse.
- **Distracting Walker2d:** All algorithms tightly overlap, as the 100k timestep horizon is too short for any method to establish stable forward kinematics in this highly brittle environment.
