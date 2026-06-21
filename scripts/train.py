import argparse
import json
import numpy as np
import torch
import os
import sys
import warnings
import pickle
import random
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from envs.distracting_cheetah import make_distracting_cheetah
from envs.distracting_hopper import make_distracting_hopper
from envs.distracting_walker import make_distracting_walker

from agents.sac import SAC, ReplayBuffer
from agents.ema_tree_sac import EMATreeSAC
from agents.l1_sac import L1SAC
from agents.group_lasso_sac import GroupLassoSAC
from agents.ema_tree_sac_reward import EMATreeSACRewardTarget

def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)
    return np.mean(returns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah', choices=['cheetah', 'hopper', 'walker2d'])
    parser.add_argument('--algo', type=str, default='sac', choices=['sac', 'ema_tree_sac', 'l1_sac', 'group_lasso_sac', 'ema_tree_sac_reward'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--ema_beta', type=float, default=0.95)
    parser.add_argument('--exp_suffix', type=str, default='')
    args = parser.parse_args()

    if args.env == 'cheetah':
        env = make_distracting_cheetah()
        eval_env = make_distracting_cheetah()
    elif args.env == 'hopper':
        env = make_distracting_hopper()
        eval_env = make_distracting_hopper()
    elif args.env == 'walker2d':
        env = make_distracting_walker()
        eval_env = make_distracting_walker()

    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    
    if args.algo == 'sac':
        agent = SAC(state_dim, env.action_space, args)
    elif args.algo == 'ema_tree_sac':
        agent = EMATreeSAC(state_dim, env.action_space, args)
    elif args.algo == 'l1_sac':
        agent = L1SAC(state_dim, env.action_space, args)
    elif args.algo == 'group_lasso_sac':
        agent = GroupLassoSAC(state_dim, env.action_space, args)
    elif args.algo == 'ema_tree_sac_reward':
        agent = EMATreeSACRewardTarget(state_dim, env.action_space, args)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    memory = ReplayBuffer(1000000)
    
    save_dir_name = args.algo + args.exp_suffix
    save_dir = os.path.join('results', str(args.timesteps), args.env, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_seed{args.seed}.pt")
    memory_path = os.path.join(save_dir, f"checkpoint_memory_seed{args.seed}.pkl")

    start_t = 0
    total_time = 0.0
    evaluations = []
    active_features_history = []

    if os.path.exists(checkpoint_path) and os.path.exists(memory_path):
        agent.load_checkpoint(checkpoint_path)
        with open(memory_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        memory.buffer = checkpoint_data['memory_buffer']
        memory.position = checkpoint_data['memory_position']
        start_t = checkpoint_data['t']
        evaluations = checkpoint_data['evaluations']
        active_features_history = checkpoint_data['active_features_history']
        total_time = checkpoint_data.get('total_time', 0.0)
        
        random.setstate(checkpoint_data['random_state'])
        np.random.set_state(checkpoint_data['np_random_state'])
        torch.set_rng_state(checkpoint_data['torch_random_state'])
        print(f"Resumed from checkpoint at timestep {start_t}")
    
    start_time = time.time() - total_time
    
    # Training Loop
    state, _ = env.reset(seed=args.seed)
    
    pbar = tqdm(range(int(args.timesteps)), desc=f"{args.env.upper()} | {args.algo.upper()} | Seed {args.seed}")
    
    # Fast forward pbar if resuming
    if start_t > 0:
        pbar.update(start_t)
        
    for t in pbar:
        if t < start_t:
            continue
            
        if t < 10000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.push(state, action, reward, next_state, terminated)
        
        state = next_state
        
        if done:
            state, _ = env.reset()
            
        if len(memory) > 1000:
            agent.update_parameters(memory, 256, 1)
            
        if (t + 1) % args.eval_freq == 0:
            avg_return = evaluate_policy(eval_env, agent)
            active = state_dim
            if hasattr(agent, 'causal_mask'):
                active = int(np.sum(agent.causal_mask))
                pbar.set_postfix({'Eval Return': f"{avg_return:.2f}", 'Features': f"{active}/{agent.original_num_inputs}"})
            elif hasattr(agent, 'get_active_features'):
                active = agent.get_active_features()
                pbar.set_postfix({'Eval Return': f"{avg_return:.2f}", 'Features': f"{active}/{agent.num_inputs}"})
            else:
                pbar.set_postfix({'Eval Return': f"{avg_return:.2f}"})
                
            evaluations.append(avg_return)
            active_features_history.append(active)
            
            # Save Checkpoint
            agent.save_checkpoint(checkpoint_path)
            with open(memory_path, 'wb') as f:
                checkpoint_data = {
                    'memory_buffer': memory.buffer,
                    'memory_position': memory.position,
                    't': t + 1,
                    'evaluations': evaluations,
                    'active_features_history': active_features_history,
                    'total_time': time.time() - start_time,
                    'random_state': random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'torch_random_state': torch.get_rng_state(),
                }
                pickle.dump(checkpoint_data, f)
            
    # Save final results
    np.save(f"{save_dir}/{args.algo}_seed{args.seed}.npy", evaluations)
    np.save(f"{save_dir}/{args.algo}_features_seed{args.seed}.npy", active_features_history)
    
    # Update live monitoring JSON
    with open(f"{save_dir}/monitoring_{save_dir_name}.json", "w") as f:
        json.dump({
            "evaluations": evaluations,
            "active_features": active_features_history,
            "wall_clock_time": time.time() - start_time
        }, f, indent=4)
        
    # Cleanup checkpoints
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(memory_path):
        os.remove(memory_path)
    if os.path.exists(checkpoint_path.replace('.pt', '_ema.pkl')):
        os.remove(checkpoint_path.replace('.pt', '_ema.pkl'))

if __name__ == '__main__':
    main()
