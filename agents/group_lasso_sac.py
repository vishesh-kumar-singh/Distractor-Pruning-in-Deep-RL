import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agents.sac import PolicyNetwork, QNetwork

class GroupLassoSAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.gl_lambda = 1e-4  # Group Lasso penalty weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        self.policy = PolicyNetwork(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
        
    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        num_inputs = state_batch.shape[1]
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Group Lasso Penalty (L2,1 norm) on first layer state features only
        gl_reg = torch.sum(torch.norm(self.critic.linear1.weight[:, :num_inputs], p=2, dim=0)) + \
                 torch.sum(torch.norm(self.critic.linear4.weight[:, :num_inputs], p=2, dim=0))
        
        qf_loss = qf_loss + self.gl_lambda * gl_reg
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        # Group Lasso Penalty on policy network input
        gl_reg_policy = torch.sum(torch.norm(self.policy.linear1.weight, p=2, dim=0))
        policy_loss = policy_loss + self.gl_lambda * gl_reg_policy
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optim.step()
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_checkpoint(self, path):
        checkpoint = {
            'critic': self.critic.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'policy': self.policy.state_dict(),
            'policy_optim': self.policy_optim.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])
