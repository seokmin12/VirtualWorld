import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    """
    A simple residual block with two linear layers and LayerNorm.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = out + residual
        return F.relu(out)


class EnhancedFeatureExtractor(nn.Module):
    """
    A feature extractor using an initial layer followed by multiple residual blocks.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.blocks(x)
        return x


class ActorCriticNetwork(nn.Module):
    """
    Shared actor-critic network with auxiliary predictors.
    - Actor head for obtaining action logits.
    - Critic head for predicting state-value.
    - Auxiliary heads for predicting balance and health (using the next-state values as targets).
    """
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.backbone = EnhancedFeatureExtractor(input_dim, hidden_dim)
        
        # Actor head: outputs logits for each action.
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic head: outputs a scalar value per state.
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Auxiliary heads: predict balance and health
        self.aux_balance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.aux_health = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: [batch, state_dim]
        features = self.backbone(x)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features).squeeze(-1)  # [batch]
        aux_balance_pred = self.aux_balance(features)
        aux_health_pred = self.aux_health(features)
        return action_logits, state_value, aux_balance_pred, aux_health_pred, features


class DeepPPOAgent:
    """
    A Deep PPO agent that uses an enhanced network architecture and
    incorporates GAE for advantage estimation along with auxiliary losses.
    
    The update() function expects rollout data including:
      - states: List[np.ndarray]
      - actions: List[int]
      - rewards: List[float]
      - dones: List[bool]
      - next_states: List[np.ndarray]
      - old_log_probs: List[float] from the policy when the actions were sampled
      - old_values: List[float] corresponding to the previous state-value estimates
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 1.0,
        entropy_coef: float = 0.01,
        aux_coef: float = 0.1,
        batch_size: int = 64,
        n_epochs: int = 10,
        max_grad_norm: float = 0.5
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.aux_coef = aux_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm

        self.network = ActorCriticNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=lr * 0.1
        )

    def select_action(self, state: np.ndarray) -> tuple:
        """
        Given a state, selects an action and returns (action, log_probability, value).
        """
        # Ensure the state has a batch dimension.
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        state_tensor = torch.FloatTensor(state).to(device)

        self.network.eval()
        with torch.no_grad():
            action_logits, state_value, _, _, _ = self.network(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(probs=action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        self.network.train()
        return int(action.item()), float(log_prob.item()), float(state_value.item())

    def compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, device=device)
        gae = 0
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
        return advantages

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        dones: List[bool],
        next_states: List[np.ndarray],
        old_log_probs: List[float],
        old_values: List[float]
    ):
        """
        Performs PPO updates using data collected from rollouts.
        """
        # Convert lists to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
        old_values_tensor = torch.FloatTensor(old_values).to(device)

        with torch.no_grad():
            # Evaluate next states to get next values
            _, next_state_values, _, _, _ = self.network(next_states_tensor)
            next_state_values = next_state_values.detach()
            # Compute advantages using GAE
            advantages = self.compute_gae(rewards_tensor, dones_tensor, old_values_tensor, next_state_values)
            returns = advantages + old_values_tensor
            # Normalize advantages using unbiased=False to avoid NaN when only one sample is present.
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dataset_size = states_tensor.shape[0]
        for _ in range(self.n_epochs):
            # Shuffle all indices for this epoch
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_next_states = next_states_tensor[batch_indices]

                action_logits, state_values, aux_balance_preds, aux_health_preds, _ = self.network(batch_states)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Ratio for PPO's clipped objective:
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (using mean squared error)
                value_loss = F.mse_loss(state_values, batch_returns)

                # Auxiliary loss: compare auxiliary predictions with ground-truth from next states.
                # Note: the state vector is assumed to have balance in index 0 and health in index 1.
                balance_targets = batch_next_states[:, 0:1]
                health_targets = batch_next_states[:, 1:2]
                aux_loss = F.mse_loss(aux_balance_preds, balance_targets) + F.mse_loss(aux_health_preds, health_targets)

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + self.aux_coef * aux_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        self.scheduler.step()

    def save(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
