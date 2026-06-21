#!/bin/bash

# Configuration
TIMESTEPS=100000 # Set to 100000 for a full run
SEEDS=(42 43 44 45 46 47 48 49 50 51)

ENVS=("cheetah" "hopper" "walker2d")

echo "Starting Multi-Environment Causal RL Benchmark Experiments"
echo "--------------------------------------------------------"

for env in "${ENVS[@]}"; do
    mkdir -p "results/${TIMESTEPS}/${env}/sac"
    mkdir -p "results/${TIMESTEPS}/${env}/ema_tree_sac"
    mkdir -p "results/${TIMESTEPS}/${env}/l1_sac"
    mkdir -p "results/${TIMESTEPS}/${env}/group_lasso_sac"
done

for env in "${ENVS[@]}"; do
    echo "========================================================"
    echo "Starting Environment: $env"
    echo "========================================================"
    
    # Phase 1: SAC and EMA-Tree SAC
    echo ">>> Phase 1: Running SAC and EMA-Tree SAC"
    for seed in "${SEEDS[@]}"; do
        echo ">>> Seed: $seed"
        for algo in "sac" "ema_tree_sac"; do
            if [ -f "results/${TIMESTEPS}/${env}/${algo}/${algo}_seed${seed}.npy" ]; then
                echo "      $algo | Already completed. Skipping."
            else
                echo "      Running $algo | Timesteps: $TIMESTEPS"
                python scripts/train.py --env "$env" --algo "$algo" --seed "$seed" --timesteps "$TIMESTEPS"
            fi
        done
    done
    echo ">>> Updating plots for Phase 1..."
    python scriecho ">>> Updating plots for Phase 1..."
    python scripts/plot_results.py --timesteps "$TIMESTEPS"
done

for env in "${ENVS[@]}"; do
    echo "========================================================"
    echo "Starting Environment: $env"
    echo "========================================================"
    
    # Phase 2: L1 SAC and Group Lasso SAC
    echo ">>> Phase 2: Running L1 SAC and Group-Lasso SAC"
    for seed in "${SEEDS[@]}"; do
        echo ">>> Seed: $seed"
        for algo in "l1_sac" "group_lasso_sac"; do
            if [ -f "results/${TIMESTEPS}/${env}/${algo}/${algo}_seed${seed}.npy" ]; then
                echo "      $algo | Already completed. Skipping."
            else
                echo "      Running $algo | Timesteps: $TIMESTEPS"
                python scripts/train.py --env "$env" --algo "$algo" --seed "$seed" --timesteps "$TIMESTEPS"
            fi
        done
    done
    echo ">>> Updating plots for Phase 2..."
    python scripts/plot_results.py --timesteps "$TIMESTEPS"
    
done

echo "--------------------------------------------------------"
echo "All main runs completed!"
echo "Check 'plots/${TIMESTEPS}' for the performance comparison."
echo "--------------------------------------------------------"

# Phase 3: Reward Target Ablation
echo "========================================================"
echo ">>> Phase 3: Reward Target Ablation (HalfCheetah)"
echo "========================================================"
mkdir -p "results/${TIMESTEPS}/cheetah/ema_tree_sac_reward"
for seed in "${SEEDS[@]}"; do
    if [ -f "results/${TIMESTEPS}/cheetah/ema_tree_sac_reward/ema_tree_sac_reward_seed${seed}.npy" ]; then
        echo "      ema_tree_sac_reward | Already completed. Skipping."
    else
        echo "      Running ema_tree_sac_reward | Timesteps: $TIMESTEPS"
        python scripts/train.py --env cheetah --algo ema_tree_sac_reward --seed "$seed" --timesteps "$TIMESTEPS"
    fi
done

# Phase 4: Full Beta Sensitivity Sweep
echo "========================================================"
echo ">>> Phase 4: Full EMA Beta Sensitivity Sweep"
echo "========================================================"
for env in "${ENVS[@]}"; do
    echo "  > Environment: $env"
    for beta in 0.5 0.1; do
        mkdir -p "results/${TIMESTEPS}/${env}/ema_tree_sac_beta${beta}"
        for seed in "${SEEDS[@]}"; do
            if [ -f "results/${TIMESTEPS}/${env}/ema_tree_sac_beta${beta}/ema_tree_sac_seed${seed}.npy" ]; then
                echo "      ema_tree_sac_beta${beta} on $env | Already completed. Skipping."
            else
                echo "      Running ema_tree_sac_beta${beta} on $env | Timesteps: $TIMESTEPS"
                python scripts/train.py --env "$env" --algo ema_tree_sac --ema_beta "$beta" --exp_suffix "_beta${beta}" --seed "$seed" --timesteps "$TIMESTEPS"
            fi
            python scripts/plot_results.py --timesteps "$TIMESTEPS"
        done
    done
done

echo "--------------------------------------------------------"
echo "Ablation Experiments Completed!"
