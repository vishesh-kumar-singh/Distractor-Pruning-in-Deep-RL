import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def smooth(scalars, weight=0.8):
    """EMA implementation for smoother learning curves."""
    if len(scalars) == 0:
        return np.array([])
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    args = parser.parse_args()

    sns.set_theme(style="darkgrid")
    
    envs = ['cheetah', 'hopper', 'walker2d']
    env_names = {'cheetah': 'Distracting HalfCheetah', 'hopper': 'Distracting Hopper', 'walker2d': 'Distracting Walker2d'}
    # Number of true state variables for each environment
    true_features = {'cheetah': 17, 'hopper': 11, 'walker2d': 17}
    
    algos = ['sac', 'l1_sac', 'group_lasso_sac', 'ema_tree_sac']
    algo_names = {'sac': 'Baseline SAC', 'l1_sac': 'L1-SAC', 'group_lasso_sac': 'Group-Lasso SAC', 'ema_tree_sac': 'EMA-Tree SAC (Ours)'}
    colors = {'sac': '#E63946', 'l1_sac': '#F4A261', 'group_lasso_sac': '#E9C46A', 'ema_tree_sac': '#2A9D8F'}
    
    eval_freq = 5000
    
    # Process data and save monitoring JSONs
    for env in envs:
        for algo in algos:
            algo_dir = os.path.join('results', str(args.timesteps), env, algo)
            if not os.path.exists(algo_dir):
                continue
                
            files = [f for f in os.listdir(algo_dir) if f.startswith(f"{algo}_seed") and f.endswith('.npy')]
            if not files:
                continue
                
            all_evals = []
            all_features = []
            for file in files:
                evals = np.load(os.path.join(algo_dir, file))
                all_evals.append(evals)
                
                feature_file = file.replace(f"{algo}_seed", f"{algo}_features_seed")
                if os.path.exists(os.path.join(algo_dir, feature_file)):
                    feats = np.load(os.path.join(algo_dir, feature_file))
                    all_features.append(feats)
                
            max_len = max([len(e) for e in all_evals])
            padded_evals = np.array([np.pad(e, (0, max_len - len(e)), constant_values=np.nan) for e in all_evals])
            
            mean_evals = np.nanmean(padded_evals, axis=0)
            std_evals = np.nanstd(padded_evals, axis=0)
            
            mean_features = []
            if all_features:
                padded_feats = np.array([np.pad(np.array(f, dtype=float), (0, max_len - len(f)), constant_values=np.nan) for f in all_features])
                mean_features = np.nanmean(padded_feats, axis=0).tolist()
            
            timesteps = np.arange(1, len(mean_evals) + 1) * eval_freq
            
            monitor_data = {
                'timesteps': timesteps.tolist(),
                'mean': mean_evals.tolist(),
                'std': std_evals.tolist(),
                'features': mean_features
            }
            with open(os.path.join(algo_dir, f'monitoring_{algo}.json'), 'w') as f:
                json.dump(monitor_data, f, indent=4)

    os.makedirs(f'plots/{args.timesteps}', exist_ok=True)
    
    for env in envs:
        # Only plot if we have data
        env_has_data = False
        for algo in algos:
            if os.path.exists(os.path.join('results', str(args.timesteps), env, algo)):
                env_has_data = True
                break
        if not env_has_data:
            continue
            
        # Generate Individual Plots instead of a grid
        fig_perf, ax_perf = plt.subplots(figsize=(8, 6))
        fig_feat, ax_feat = plt.subplots(figsize=(8, 6))
        
        for algo in algos:
            algo_dir = os.path.join('results', str(args.timesteps), env, algo)
            json_path = os.path.join(algo_dir, f'monitoring_{algo}.json')
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Need to check if it's the aggregated JSON from plot_results or raw JSON from train.py
            if 'mean' not in data:
                continue
                
            timesteps = np.array(data['timesteps'])
            mean_evals = np.array(data['mean'])
            std_evals = np.array(data['std'])
            
            # Smooth the curves to prevent visual clutter
            smoothed_mean = smooth(mean_evals, weight=0.6)
            smoothed_std = smooth(std_evals, weight=0.6)
            
            # Plot Learning Curve on the combined axis
            ax_perf.plot(timesteps, smoothed_mean, label=algo_names[algo], color=colors[algo], linewidth=2.5)
            ax_perf.fill_between(timesteps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors[algo], alpha=0.15)
            
            # Plot Feature Curve
            if 'features' in data and data['features']:
                features = np.array(data['features'])
                ax_feat.plot(timesteps, features, label=algo_names[algo], color=colors[algo], linewidth=2.5)
                
        # Format Performance Plot
        ax_perf.set_title(f'Performance on {env_names[env]}', fontsize=16, fontweight='bold')
        ax_perf.set_xlabel('Timesteps', fontsize=14)
        ax_perf.set_ylabel('Average Return', fontsize=14)
        ax_perf.tick_params(axis='both', which='major', labelsize=12)
        if ax_perf.get_legend_handles_labels()[0]:
            ax_perf.legend(fontsize=12, loc='upper left')
        
        # Format Feature Plot
        ax_feat.axhline(y=true_features[env], color='black', linestyle='--', linewidth=2, label=f'True Features ({true_features[env]})')
        ax_feat.set_title(f'Feature Dynamics ({env_names[env]})', fontsize=16, fontweight='bold')
        ax_feat.set_xlabel('Timesteps', fontsize=14)
        ax_feat.set_ylabel('Active Features', fontsize=14)
        ax_feat.tick_params(axis='both', which='major', labelsize=12)
        if ax_feat.get_legend_handles_labels()[0]:
            ax_feat.legend(fontsize=12, loc='upper right')
        
        fig_perf.tight_layout()
        fig_perf.savefig(f'plots/{args.timesteps}/{env}_performance.pdf', format='pdf', dpi=600, bbox_inches='tight')
        
        fig_feat.tight_layout()
        fig_feat.savefig(f'plots/{args.timesteps}/{env}_features.pdf', format='pdf', dpi=600, bbox_inches='tight')
        
        plt.close(fig_perf)
        plt.close(fig_feat)
        
        print(f"Saved individual plots for {env} to plots/{args.timesteps}/")

if __name__ == '__main__':
    main()
