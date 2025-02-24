import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = out + residual
        return F.relu(out)


class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        return x


class Network(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_entities: int):
        super().__init__()
        self.num_entities = num_entities
        self.backbone = EnhancedFeatureExtractor(input_dim, hidden_dim)

        # 멀티 헤드 액터 (엔티티별 독립적 행동 결정)
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_entities)
        ])

        # 크리틱 헤드 (전체 상태 가치 평가)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 보조 예측기 출력 차원 수정 (num_entities → 1)
        self.aux_balance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)  # 단일 값 예측
        )
        self.aux_health = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)  # 단일 값 예측
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        entity_logits = [head(features) for head in self.actor_heads]
        state_value = self.critic_head(features).squeeze(-1)
        aux_balance_pred = self.aux_balance(features).squeeze(-1)  # [B, 1] → [B]
        aux_health_pred = self.aux_health(features).squeeze(-1)    # [B, 1] → [B]
        return entity_logits, state_value, aux_balance_pred, aux_health_pred, features


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
            num_entities: int = 2,
            lr: float = 3e-4,
            gamma: float = 0.999,
            gae_lambda: float = 0.97,
            clip_range: float = 0.2,
            value_coef: float = 0.5,
            entropy_coef: float = 0.1,
            aux_coef: float = 0.2,
            batch_size: int = 128,
            n_epochs: int = 10,
            max_grad_norm: float = 0.5
    ):
        self.num_entities = num_entities
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

        self.network = Network(state_dim, hidden_dim, action_dim, num_entities).to(device)
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
            entity_logits, state_value, _, _, _ = self.network(state_tensor)
            actions = []
            log_probs = []
            for logits in entity_logits:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(action.item())
                log_probs.append(log_prob.item())
        self.network.train()
        return actions, log_probs, float(state_value.item())

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
                batch_indices = indices[start: start + self.batch_size]
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_next_states = next_states_tensor[batch_indices]

                entity_logits, state_values, aux_balance_preds, aux_health_preds, _ = self.network(batch_states)
                dists = [Categorical(logits=logits) for logits in entity_logits]
                new_log_probs = [dist.log_prob(action) for dist, action in zip(dists, batch_actions)]
                entropies = [dist.entropy().mean() for dist in dists]

                # Ratio for PPO's clipped objective:
                ratios = [torch.exp(new_log_prob - old_log_prob) for new_log_prob, old_log_prob in
                          zip(new_log_probs, batch_old_log_probs)]
                surr1 = [ratio * advantage for ratio, advantage in zip(ratios, batch_advantages)]
                surr2 = [torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage for ratio, advantage
                         in zip(ratios, batch_advantages)]
                policy_loss = -sum(
                    [torch.min(surr1_i, surr2_i).mean() for surr1_i, surr2_i in zip(surr1, surr2)]) / self.num_entities

                # Value loss (using mean squared error)
                value_loss = F.mse_loss(state_values, batch_returns)

                # 보조 타겟 인덱스 수정 (단일 엔티티 기준)
                balance_targets = batch_next_states[:, 0]  # 밸런스: 0번 인덱스
                health_targets = batch_next_states[:, 1]   # 건강: 1번 인덱스
                
                # 차원 일치화 (squeeze 제거)
                aux_loss = F.mse_loss(aux_balance_preds, balance_targets) + \
                           F.mse_loss(aux_health_preds, health_targets)

                # 엔티티별 엔트로피 계산 방식 개선
                entropy = torch.stack([dist.entropy().mean() for dist in dists]).mean()

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