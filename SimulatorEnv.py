import gym
import numpy as np
from module.Bank import Bank
from module.Entity import Entity


class Env(gym.Env):
    def __init__(self):
        self.bank = Bank()
        self.entity = Entity("entity")
        self.entity.account = self.bank.createAccount(self.entity.name)
        self.action_space = gym.spaces.Discrete(3)  # mine, rest, leisure
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.current_day: int = 0
        self.current_hour: float = 0.0
        self.work_hours: float = 0.0

    def get_state(self) -> np.ndarray:
        condition_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        norm_balance: float = self.entity.account.balance / 10000.0
        norm_health: float = self.entity.health / 100.0
        norm_happiness: float = self.entity.happiness / 1.0
        condition_value: float = condition_map.get(self.entity.current_condition)
        norm_time_of_day: float = self.current_day / 24.0
        norm_work_hours: float = self.work_hours / 12.0
        norm_age: float = self.entity.age / 100.0

        state = np.array([
            norm_balance,
            norm_health,
            norm_happiness,
            condition_value,
            norm_time_of_day,
            norm_work_hours,
            norm_age
        ], dtype=np.float32)

        # Add clipping to ensure all values are within bounds
        state = np.clip(state, 0.0, 1.0)

        # Add check for NaN values
        if np.any(np.isnan(state)):
            print("Warning: NaN values detected in state!")
            state = np.nan_to_num(state, nan=0.0)

        return state

    def step(self, action: int):
        reward: float = 0.0
        done: bool = False  # termination flag

        # Check if the day has ended, then update time/date and apply daily penalties.
        if self.current_hour >= 24:
            if self.work_hours > 12:
                self.entity.health -= 10  # penalty for overworking in a day
            self.current_day += 1
            # Increase age every 365 days.
            if self.current_day % 365 == 0:
                self.entity.age += 1
            self.current_hour = 0
            self.work_hours = 0

        # Execute the chosen action.
        if action == 0:  # Mine
            reward += self.entity.mine()
            self.work_hours += 1  # increase work hours when mining
        elif action == 1:  # Rest
            reward += self.entity.rest()
        elif action == 2:  # Leisure
            reward += self.entity.leisure()
        else:
            reward -= 1  # penalty on unknown action

        # Survival reward: Encourage agent to keep health and happiness high.
        survival_reward = 0.5 * (self.entity.health / 100.0) + 0.5 * (self.entity.happiness)
        reward += survival_reward

        # Move forward in time and incur a slight natural decrease in happiness.
        self.current_hour += 1
        self.entity.happiness = max(0.0, self.entity.happiness - 0.005)

        # Termination Conditions: health depleted, age exceeds lifespan, or happiness too low.
        if self.entity.health <= 0 or self.entity.age >= self.entity.lifespan or self.entity.happiness < 0.05:
            done = True

        info = {
            'current_day': self.current_day,
            'work_hours': self.work_hours,
            'health': self.entity.health,
            'balance': self.entity.account.balance,
            'happiness': self.entity.happiness
        }

        return self.get_state(), reward, done, False, info

    def reset(self):
        self.entity = Entity("entity")
        self.entity.account = self.bank.createAccount(self.entity.name)
        self.current_day = 0
        self.current_hour = 0
        self.work_hours = 0
        return self.get_state(), {}
