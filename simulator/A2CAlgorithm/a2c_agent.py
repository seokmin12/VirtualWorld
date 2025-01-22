import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class A2CNetwork(nn.Module):
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

class A2CAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = A2CNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
    def choose_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.network(state)
        
        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
        else:
            action = torch.argmax(action_probs)
            
        return action.item()
    
    def train(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 현재 상태의 행동 확률과 가치
        action_probs, state_values = self.network(states)
        dist = Categorical(action_probs)
        
        # 다음 상태의 가치 계산
        with torch.no_grad():
            _, next_state_values = self.network(next_states)
            next_state_values = next_state_values.squeeze()
            
        # Advantage 계산
        returns = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = returns - state_values.squeeze()
        
        # Actor loss (정책 손실)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (가치 손실)
        critic_loss = advantages.pow(2).mean()
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
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