import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from Simulator import Simulator
from model.Entity import Entity

class SimulatorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.simulator = Simulator()
        self.entity = Entity("Agent")
        self.simulator.user = self.entity
        self.simulator.user.account = self.simulator.bank.createAccount(self.simulator.user.name)

        # 행동 공간: [채굴, 휴식, 거래]
        self.action_space = spaces.Discrete(3)

        # 관찰 공간: [체력, 컨디션, 채굴력, 위험감수성, 작업윤리, 거래기술, 계좌잔액, 시간]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, np.inf, 24]),
            dtype=np.float32
        )

    def _get_state(self):
        entity_state = self.entity.get_state_vector()
        return np.array(entity_state + [
            self.simulator.user.account.balance / 10000,  # 정규화된 잔액
            self.simulator.time / (24 * 3600)  # 정규화된 시간
        ], dtype=np.float32)

    def step(self, action):
        success_prob = random.random()
        action_success = False
        
        if action == 0:  # 채굴
            success_threshold = 0.5 + (self.entity.work_ethic * 0.3)
            action_success = success_prob < success_threshold
            if action_success:
                self.simulator.mine(self.entity)
                self.entity.total_mined += 1
            else:
                self.entity.health -= 0.1
                
        elif action == 1:  # 휴식
            condition_values = {'low': 0, 'medium': 1, 'high': 2}
            condition_value = condition_values[self.entity.current_condition]
            recovery_bonus = 0.1 + (condition_value * 0.2)
            self.entity.rest_recovery_rate *= (1 + recovery_bonus)
            self.simulator.rest(self.entity)
            self.entity.rest_recovery_rate /= (1 + recovery_bonus)
            action_success = True
            
        elif action == 2:  # 거래
            success_threshold = 0.4 + (self.entity.trading_skill * 0.4)
            action_success = success_prob < success_threshold
            if action_success:
                trade_profit = random.uniform(100, 1000) * (1 + self.entity.trading_skill)
                self.entity.account.balance += trade_profit
                self.entity.total_traded += 1
            else:
                loss = random.uniform(50, 500) * (1 - self.entity.trading_skill * 0.5)
                self.entity.account.balance = max(0, self.entity.account.balance - loss)
            
        state = self._get_state()
        reward = self._calculate_reward(action, action_success)
        
        terminated = (self.entity.health <= 0 or 
                     self.simulator.time >= 3600 * 24 or
                     self.entity.account.balance >= 1000000)
        truncated = False  # 에피소드가 외부 요인으로 인해 중단된 경우
                
        return state, reward, terminated, truncated, {"success": action_success}

    def _calculate_reward(self, action, success):
        base_reward = 0
        balance_reward = np.log1p(self.entity.account.balance) * 0.01
        
        action_reward = 0
        if action == 0:  # 채굴
            action_reward = 20 if success else -10
            action_reward *= (1 + self.entity.work_ethic)
        elif action == 1:  # 휴식
            action_reward = 5 if self.entity.health < 0.5 else -5
        elif action == 2:  # 거래
            action_reward = 30 if success else -20
            action_reward *= (1 + self.entity.trading_skill)
        
        health_penalty = -50 if self.entity.health < 0.2 else 0
        time_efficiency = 10 if self.simulator.time < 3600 * 12 else 0
        
        total_reward = (
            balance_reward + 
            action_reward + 
            health_penalty + 
            time_efficiency
        )
        
        risk_modifier = 1 + (self.entity.risk_tolerance - 0.5) * 0.4
        total_reward *= risk_modifier
        
        return total_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium 표준을 따르기 위한 seed 설정
        
        self.simulator = Simulator()
        self.entity = Entity("Agent")
        self.simulator.user = self.entity
        self.simulator.user.account = self.simulator.bank.createAccount(self.simulator.user.name)
        
        return self._get_state(), {}  # 상태와 추가 정보를 반환 