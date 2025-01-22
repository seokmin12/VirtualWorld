import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95,
                 clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, update_epochs=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
    def choose_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.network(state)
        
        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()
        else:
            action = torch.argmax(action_probs)
            return action.item()
    
    def train(self, states, actions, old_log_probs, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # GAE(Generalized Advantage Estimation) 계산
        with torch.no_grad():
            _, next_values = self.network(next_states)
            next_values = next_values.squeeze()
            returns = rewards + self.gamma * next_values * (1 - dones)
            
        for _ in range(self.update_epochs):
            # 현재 정책의 행동 확률과 상태 가치
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # ratio = exp(log(π_new) - log(π_old))
            ratios = torch.exp(curr_log_probs - old_log_probs)
            
            # Advantage 계산
            advantages = returns - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO 클립 목적 함수
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # 전체 손실
            total_loss = (
                actor_loss 
                + self.value_coef * value_loss 
                - self.entropy_coef * entropy.mean()
            )
            
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 