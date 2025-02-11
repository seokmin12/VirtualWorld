import random
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

        self.mining_difficulty = 10000  # Mining difficulty
        self.reward_per_block = 50  # Reward per successful mining block

    def mine(self) -> float:
        mining_time = self.mining_difficulty / (self.mining_power * 10)
        health_cost = mining_time * 0.05
        self.health -= health_cost
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
