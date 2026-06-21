import numpy as np
import torch
from agents.ema_tree_sac import EMATreeSAC

class EMATreeSACRewardTarget(EMATreeSAC):
    def discover_causal_mask(self, memory):
        if len(memory) < 1000:
            return self.causal_mask
            
        from sklearn.ensemble import RandomForestRegressor
        
        state, action, reward, next_state, _ = memory.sample(1000)
        
        # Inject 5 dummy noise variables for a stronger noise floor
        dummy_noise = np.random.randn(state.shape[0], 5).astype(np.float32)
        X = np.concatenate([state, action, dummy_noise], axis=1)
        
        # THE ABLATION: Target the immediate reward instead of the Q-value!
        Y = reward.flatten()
        
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
        new_mask = (self.ema_importances > self.ema_dummy + 1e-4).astype(np.float32)
            
        return new_mask
