import gymnasium as gym
import numpy as np

class DistractingHalfCheetah(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Original shape is (17,)
        orig_shape = self.observation_space.shape[0]
        self.num_distractors = 10
        new_shape = (orig_shape + self.num_distractors,)
        
        # We need to expand the observation space bounds
        high = np.inf * np.ones(new_shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        
    def observation(self, obs):
        # Increase noise magnitude to make it harder for standard SAC
        noise = np.random.normal(0, 0.5, size=(5,))
        # Pure noise distractors with high variance
        pure_noise = np.random.normal(0, 5.0, size=(5,))
        # Spurious correlations based on true state variables, heavily scaled
        spurious = np.array([
            obs[8] * 2.5 + noise[0],
            obs[0] * obs[1] * 5.0 + noise[1],
            np.sin(obs[2]) * 5.0 + noise[2],
            (obs[3] + obs[4]) * 5.0 + noise[3],
            np.exp(-np.abs(obs[8])) * 5.0 + noise[4]
        ])
        
        distractors = np.concatenate([pure_noise, spurious], axis=0).astype(np.float32)
        return np.concatenate([obs, distractors], axis=0)

def make_distracting_cheetah():
    env = gym.make("HalfCheetah-v5")
    return DistractingHalfCheetah(env)
