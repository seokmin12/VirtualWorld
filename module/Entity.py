import random
import numpy as np
from typing import List


class Entity:
    def __init__(self, name: str, bank):
        self.name: str = name
        self.health: float = 100.0
        self.age: int = 20
        self.lifespan: int = 100
        self.happiness: float = 1.0
        # Condition represented numerically internally: low = 0, medium = 0.5, high = 1.0
        self.current_condition: str = "medium"  # Valid values: "low", "medium", "high"
        self.mining_power: float = 10.0
        self.rest_recovery: float = 10.0
        # Behavioral traits
        self.total_mined: int = 0
        self.total_traded: int = 0
        self.account = bank.createAccount(self.name)  # Will be assigned via the Bank

        self.mining_difficulty: int = 10000  # Mining difficulty
        self.reward_per_block: int = 50  # Reward per successful mining block

        # religion
        self.religion = random.choice(['Christianity', 'Buddhism', 'Catholic', 'Atheist'])  # 기독교, 불교, 천주교, 무교
        self.religion_power: float = 0.0

        self.current_day: int = 0
        self.current_hour: float = 0.0
        self.work_hours: float = 0.0
        # 플래그 추가: 이미 종료되었으면 True
        self.terminated: bool = False

    def reset(self):
        self.health = 100.0
        self.age = 20
        self.lifespan = 100
        self.happiness = 1.0
        self.current_condition = "medium"
        self.total_mined = 0
        self.total_traded = 0
        self.religion_power = 0.0
        self.current_day = 0
        self.current_hour = 0.0
        self.work_hours = 0.0

    def get_state(self) -> np.ndarray:
        condition_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        norm_balance: float = self.account.balance / 10000.0
        norm_health: float = self.health / 100.0
        norm_happiness: float = self.happiness / 1.0
        condition_value: float = condition_map.get(self.current_condition)
        norm_time_of_day: float = self.current_day / 24.0
        norm_work_hours: float = self.work_hours / 12.0
        norm_age: float = self.age / 100.0
        norm_religion_power: float = self.religion_power / 100.0

        state = np.array([
            norm_balance,
            norm_health,
            norm_happiness,
            condition_value,
            norm_time_of_day,
            norm_work_hours,
            norm_age,
            norm_religion_power
        ], dtype=np.float32)

        # Add clipping to ensure all values are within bounds
        state = np.clip(state, 0.0, 1.0)

        # Add check for NaN values
        if np.any(np.isnan(state)):
            print("Warning: NaN values detected in state!")
            state = np.nan_to_num(state, nan=0.0)

        return state

    def mine(self) -> float:
        mining_time = self.mining_difficulty / (self.mining_power * 10)
        health_cost = mining_time * 0.05
        self.health = max(0.0, self.health - health_cost)
        self.happiness = min(1.0, self.happiness - 0.05)
        self.total_mined += 1
        if self.account:
            self.account.deposit(self.reward_per_block)
        reward = 1.0

        return reward

    def rest(self) -> float:
        recovery = self.rest_recovery
        self.health = min(100, self.health + recovery)
        self.happiness = min(1.0, self.happiness - 0.05)
        # Improve condition with a certain chance.
        self.current_condition = "high"
        reward: float = 1.0
        return reward

    def leisure(self) -> float:
        self.happiness = min(1.0, self.happiness + 0.1)
        self.health -= 7
        reward: float = 0.8
        return reward

    def religious_activity(self) -> float:
        if self.religion == "Atheist":
            self.happiness = max(0.0, self.happiness - 0.05)
            reward = 0.5
        else:
            self.happiness = min(1.0, self.happiness + 0.1)
            self.religion_power = min(100.0, self.religion_power + 0.4)
            self.health = max(0.0, self.health - 2)
            reward = 1.2

        return reward

    def trade(self, other, amount: float) -> float:
        if amount <= 0:
            return -1.0

        if self.account.balance < amount:
            return -1.0

        if not self.account.withdraw(amount):
            return -1.0

        other.account.deposit(amount)

        self.total_traded += amount
        other.total_traded += amount

        self.happiness = min(1.0, self.happiness + 0.02)
        other.happiness = min(1.0, other.happiness + 0.02)

        return 1.0
