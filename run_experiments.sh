#!/bin/bash

# Configuration
TIMESTEPS=1000000 # Set to 100000 for a full run
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
    python scripts/plot_results.py --timesteps "$TIMESTEPS"
    
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
echo "All runs completed!"
echo "Check 'plots/${TIMESTEPS}' for the performance comparison."
