import numpy as np
import torch
import torch.nn.functional as F
from agents.sac import SAC

class EMATreeSAC(SAC):
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        self.original_num_inputs = num_inputs
        self.causal_mask = np.ones(num_inputs, dtype=np.float32)
        self.causal_mask_tensor = torch.FloatTensor(self.causal_mask).to(self.device)
        self.ema_importances = None
        self.ema_alpha = 0.1
        self.discovery_freq = 5000
        self.updates_done = 0
        
    def discover_causal_mask(self, memory):
        if len(memory) < 1000:
            return self.causal_mask
            
        from sklearn.ensemble import RandomForestRegressor
        
        state, action, reward, next_state, _ = memory.sample(1000)
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        with torch.no_grad():
            q1, q2 = self.critic(state_tensor * self.causal_mask_tensor, action_tensor)
            q_val = torch.min(q1, q2).cpu().numpy()
            
        # Inject 5 dummy noise variables for a stronger noise floor
        dummy_noise = np.random.randn(state.shape[0], 5).astype(np.float32)
        X = np.concatenate([state, action, dummy_noise], axis=1)
        Y = q_val
        
        # Decorrelated RF
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, max_features='sqrt', n_jobs=-1, random_state=42)
        rf.fit(X, Y)
        
        feature_importance = rf.feature_importances_
        state_importance = feature_importance[:self.original_num_inputs]
        dummy_importances = feature_importance[-5:]
        dummy_importance = np.mean(dummy_importances)
        
        # EMA Update
        if self.ema_importances is None:
            self.ema_importances = state_importance
            self.ema_dummy = dummy_importance
        else:
            self.ema_importances = self.ema_alpha * state_importance + (1 - self.ema_alpha) * self.ema_importances
            self.ema_dummy = self.ema_alpha * dummy_importance + (1 - self.ema_alpha) * self.ema_dummy
            
        # The EMA mathematically smooths out the noise over time. 
        # By comparing the smoothed state importances to the smoothed dummy importance,
        # we have a perfectly stable, dynamic statistical threshold.
        new_mask = (self.ema_importances > self.ema_dummy + 1e-4).astype(np.float32)
            
        return new_mask
        
    def update_parameters(self, memory, batch_size, updates):
        self.updates_done += 1
        
        if self.updates_done % self.discovery_freq == 0:
            computed_mask = self.discover_causal_mask(memory)
            
            # Apply mask from the beginning (after initial discovery at step 1000)
            if self.updates_done >= 0:
                self.causal_mask = computed_mask
                
            self.causal_mask_tensor = torch.FloatTensor(self.causal_mask).to(self.device)
        
        # Standard SAC Update but with masked states
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)
        
        state_batch = state_batch * self.causal_mask
        next_state_batch = next_state_batch * self.causal_mask
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def select_action(self, state, evaluate=False):
        masked_state = state * self.causal_mask
        return super().select_action(masked_state, evaluate)

    def save_checkpoint(self, path):
        import pickle
        super().save_checkpoint(path)
        ema_path = path.replace('.pt', '_ema.pkl')
        ema_data = {
            'causal_mask': self.causal_mask,
            'ema_importances': self.ema_importances,
            'ema_dummy': getattr(self, 'ema_dummy', None),
            'updates_done': self.updates_done
        }
        with open(ema_path, 'wb') as f:
            pickle.dump(ema_data, f)
            
    def load_checkpoint(self, path):
        import os
        import pickle
        super().load_checkpoint(path)
        ema_path = path.replace('.pt', '_ema.pkl')
        if os.path.exists(ema_path):
            with open(ema_path, 'rb') as f:
                ema_data = pickle.load(f)
            self.causal_mask = ema_data['causal_mask']
            self.causal_mask_tensor = torch.FloatTensor(self.causal_mask).to(self.device)
            self.ema_importances = ema_data['ema_importances']
            if ema_data.get('ema_dummy') is not None:
                self.ema_dummy = ema_data['ema_dummy']
            self.updates_done = ema_data['updates_done']
