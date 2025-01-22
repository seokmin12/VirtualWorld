import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        # He 초기화
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

        # Layer Normalization 사용 (배치 크기에 독립적)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(32)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(torch.relu(self.ln1(self.fc1(x))))
        x = self.dropout(torch.relu(self.ln2(self.fc2(x))))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # 할인율
        self.epsilon = 1.0  # 탐험률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.batch_size = 64
        self.update_target_freq = 10  # 타겟 네트워크 업데이트 주기
        self.prioritized_replay = True
        self.priorities = deque(maxlen=10000)
        self.alpha = 0.6  # 우선순위 계수
        self.beta = 0.4  # 중요도 샘플링 계수
        self.beta_increment = 0.001

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # 초기 우선순위를 최대값으로 설정
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        if self.prioritized_replay:
            probs = np.array(self.priorities) ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), batch_size, p=probs)
            
            # 중요도 샘플링 가중치 계산
            weights = (len(self.memory) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.FloatTensor(weights).to(self.device)
            
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            indices = np.random.choice(len(self.memory), batch_size)
            weights = torch.ones(batch_size).to(self.device)

        batch = [self.memory[idx] for idx in indices]
        
        states = torch.FloatTensor([i[0] for i in batch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in batch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in batch]).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # TD 오차 계산 및 우선순위 업데이트
        td_errors = torch.abs(target_q_values - current_q_values.squeeze()).detach().cpu().numpy()
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error

        # 가중치가 적용된 손실 계산
        loss = (weights * (target_q_values - current_q_values.squeeze()) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 그래디언트 클리핑
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
