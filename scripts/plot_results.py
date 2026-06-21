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
    colors = {'sac': '#E63946', 'l1_sac': '#F4A261', 'group_lasso_sac': '#9B5DE5', 'ema_tree_sac': '#2A9D8F'}
    
    # Sensitivity Sweep Configuration
    sensitivity_algos = ['ema_tree_sac_beta0.1', 'ema_tree_sac_beta0.5', 'ema_tree_sac_beta0.95']
    sensitivity_names = {'ema_tree_sac_beta0.1': r'$\beta=0.1$ (Fast)', 'ema_tree_sac_beta0.5': r'$\beta=0.5$ (Medium)', 'ema_tree_sac_beta0.95': r'$\beta=0.95$ (Slow)'}
    sensitivity_colors = {'ema_tree_sac_beta0.1': '#E63946', 'ema_tree_sac_beta0.5': '#F4A261', 'ema_tree_sac_beta0.95': '#2A9D8F'}
    
    all_process_algos = algos + sensitivity_algos + ['ema_tree_sac_reward']
    
    eval_freq = 5000
    
    # Process data and save monitoring JSONs
    for env in envs:
        for algo in all_process_algos:
            algo_dir = os.path.join('results', str(args.timesteps), env, algo)
            if not os.path.exists(algo_dir):
                continue
                
            base_algo = algo.split('_beta')[0] if '_beta' in algo else algo
            files = [f for f in os.listdir(algo_dir) if f.startswith(f"{base_algo}_seed") and f.endswith('.npy')]
            if not files:
                continue
                
            all_evals = []
            all_features = []
            for file in files:
                evals = np.load(os.path.join(algo_dir, file))
                all_evals.append(evals)
                
                feature_file = file.replace(f"{base_algo}_seed", f"{base_algo}_features_seed")
                if os.path.exists(os.path.join(algo_dir, feature_file)):
                    feats = np.load(os.path.join(algo_dir, feature_file))
                    all_features.append(feats)
                
            max_len = max([len(e) for e in all_evals])
            padded_evals = np.array([np.pad(e, (0, max_len - len(e)), constant_values=np.nan) for e in all_evals])
            
            mean_evals = np.nanmean(padded_evals, axis=0)
            std_evals = np.nanstd(padded_evals, axis=0) / np.sqrt(len(all_evals)) # Use Standard Error for cleaner plots
            
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
        # Loop over beta values to generate baseline comparison plots for EACH beta
        for beta in [0.1, 0.5, 0.95]:
            beta_algo = f'ema_tree_sac_beta{beta}'
            plot_algos = ['sac', 'l1_sac', 'group_lasso_sac', beta_algo]
            
            # Dynamically add the specific beta variant to the dictionaries
            current_names = algo_names.copy()
            current_names[beta_algo] = f'EMA-Tree SAC (Beta {beta})'
            current_colors = colors.copy()
            current_colors[beta_algo] = '#2A9D8F'
            
            # Only plot if we have data for this environment
            env_has_data = False
            for algo in plot_algos:
                if os.path.exists(os.path.join('results', str(args.timesteps), env, algo)):
                    env_has_data = True
                    break
            if not env_has_data:
                continue
                
            # Generate Baseline Comparison Plots for this specific Beta
            fig_perf, ax_perf = plt.subplots(figsize=(8, 6))
            fig_feat, ax_feat = plt.subplots(figsize=(8, 6))
            
            for algo in plot_algos:
                algo_dir = os.path.join('results', str(args.timesteps), env, algo)
                json_path = os.path.join(algo_dir, f'monitoring_{algo}.json')
                if not os.path.exists(json_path):
                    continue
                    
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                if 'mean' not in data:
                    continue
                    
                timesteps = np.array(data['timesteps'])
                mean_evals = np.array(data['mean'])
                std_evals = np.array(data['std'])
                
                # Smooth the curves to prevent visual clutter
                smoothed_mean = smooth(mean_evals, weight=0.6)
                smoothed_std = smooth(std_evals, weight=0.6)
                
                # Plot Learning Curve on the combined axis
                ax_perf.plot(timesteps, smoothed_mean, label=current_names[algo], color=current_colors[algo], linewidth=2.5)
                ax_perf.fill_between(timesteps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=current_colors[algo], alpha=0.15)
                
                # Plot Feature Curve
                if 'features' in data and data['features']:
                    features = np.array(data['features'])
                    ax_feat.plot(timesteps, features, label=current_names[algo], color=current_colors[algo], linewidth=2.5)
                    
            # Format Performance Plot
            ax_perf.set_title(f'Baseline Comparison ({env_names[env]}, Beta={beta})', fontsize=21, fontweight='bold')
            ax_perf.set_xlabel('Timesteps', fontsize=19)
            ax_perf.set_ylabel('Average Return', fontsize=19)
            ax_perf.tick_params(axis='both', which='major', labelsize=15)
            if ax_perf.get_legend_handles_labels()[0]:
                ax_perf.legend(fontsize=15, loc='upper left')
            
            # Format Feature Plot
            ax_feat.axhline(y=true_features[env], color='black', linestyle='--', linewidth=2, label=f'True Features ({true_features[env]})')
            ax_feat.set_title(f'Feature Dynamics ({env_names[env]}, Beta={beta})', fontsize=21, fontweight='bold')
            ax_feat.set_xlabel('Timesteps', fontsize=19)
            ax_feat.set_ylabel('Active Features', fontsize=19)
            ax_feat.tick_params(axis='both', which='major', labelsize=15)
            if ax_feat.get_legend_handles_labels()[0]:
                ax_feat.legend(fontsize=14, loc='upper right')
            
            fig_perf.tight_layout()
            fig_perf.savefig(f'plots/{args.timesteps}/{env}_baseline_comparison_beta{beta}_performance.pdf', format='pdf', dpi=600, bbox_inches='tight')
            
            fig_feat.tight_layout()
            fig_feat.savefig(f'plots/{args.timesteps}/{env}_baseline_comparison_beta{beta}_features.pdf', format='pdf', dpi=600, bbox_inches='tight')
            
            plt.close(fig_perf)
            plt.close(fig_feat)
            
            print(f"Saved baseline comparison plots for {env} (Beta={beta}) to plots/{args.timesteps}/")
        
    # Generate Sensitivity Plots
    for env in envs:
        env_has_data = False
        for algo in sensitivity_algos:
            if os.path.exists(os.path.join('results', str(args.timesteps), env, algo)):
                env_has_data = True
                break
        if not env_has_data:
            continue
            
        fig_sens, ax_sens = plt.subplots(figsize=(8, 6))
        fig_sens_feat, ax_sens_feat = plt.subplots(figsize=(8, 6))
        
        for algo in sensitivity_algos:
            algo_dir = os.path.join('results', str(args.timesteps), env, algo)
            json_path = os.path.join(algo_dir, f'monitoring_{algo}.json')
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if 'mean' not in data:
                continue
                
            timesteps = np.array(data['timesteps'])
            mean_evals = np.array(data['mean'])
            std_evals = np.array(data['std'])
            
            smoothed_mean = smooth(mean_evals, weight=0.6)
            smoothed_std = smooth(std_evals, weight=0.6)
            
            ax_sens.plot(timesteps, smoothed_mean, label=sensitivity_names[algo], color=sensitivity_colors[algo], linewidth=2.5)
            ax_sens.fill_between(timesteps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=sensitivity_colors[algo], alpha=0.15)
            
            if 'features' in data and data['features']:
                features = np.array(data['features'])
                ax_sens_feat.plot(timesteps, features, label=sensitivity_names[algo], color=sensitivity_colors[algo], linewidth=2.5)
            
        ax_sens.set_title(f'Hyperparameter Sensitivity ({env_names[env]})', fontsize=21, fontweight='bold')
        ax_sens.set_xlabel('Timesteps', fontsize=19)
        ax_sens.set_ylabel('Average Return', fontsize=19)
        ax_sens.tick_params(axis='both', which='major', labelsize=15)
        if ax_sens.get_legend_handles_labels()[0]:
            ax_sens.legend(fontsize=15, loc='upper left')
            
        ax_sens_feat.axhline(y=true_features[env], color='black', linestyle='--', linewidth=2, label=f'True Features ({true_features[env]})')
        ax_sens_feat.set_title(f'Pruning Speed vs EMA Beta ({env_names[env]})', fontsize=21, fontweight='bold')
        ax_sens_feat.set_xlabel('Timesteps', fontsize=19)
        ax_sens_feat.set_ylabel('Active Features', fontsize=19)
        ax_sens_feat.tick_params(axis='both', which='major', labelsize=15)
        if ax_sens_feat.get_legend_handles_labels()[0]:
            ax_sens_feat.legend(fontsize=14, loc='upper right')
        
        fig_sens.tight_layout()
        fig_sens.savefig(f'plots/{args.timesteps}/{env}_sensitivity_performance.pdf', format='pdf', dpi=600, bbox_inches='tight')
        
        fig_sens_feat.tight_layout()
        fig_sens_feat.savefig(f'plots/{args.timesteps}/{env}_sensitivity_features.pdf', format='pdf', dpi=600, bbox_inches='tight')
        
        plt.close(fig_sens)
        plt.close(fig_sens_feat)
        print(f"Saved sensitivity plots for {env} to plots/{args.timesteps}/")

if __name__ == '__main__':
    main()
