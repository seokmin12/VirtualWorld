import gym
import numpy as np
from model.Bank import Bank
from model.Entity import Entity


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
        norm_mining_power: float = self.entity.mining_power / 100.0
        condition_value: float = condition_map.get(self.entity.current_condition, 0.5)
        norm_time_of_day: float = self.current_day / 24.0
        norm_work_hours: float = self.work_hours / 12.0
        norm_age: float = self.entity.age / 100.0

        state = np.array([
            norm_balance,
            norm_health,
            norm_mining_power,
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
        done: bool = False  # Initialize done variable

        # 하루의 끝을 확인하고 날짜와 시간을 업데이트
        if self.current_hour >= 24:
            if self.work_hours > 12:
                self.entity.health -= 10  # overworking에 대한 패널티
            self.current_day += 1
            # 365일이 지날 때마다 나이 1살 증가
            if self.current_day % 365 == 0:
                self.entity.age += 1
            self.current_hour = 0
            self.work_hours = 0

        if action == 0:  # Mine
            reward += self.entity.mine()
        elif action == 1:  # Rest
            reward += self.entity.rest()
        elif action == 2:  # Leisure
            reward += self.entity.leisure()
        else:
            reward += -1

        # 시뮬레이션 시간을 증가시키고 약간의 행복도 감소를 적용
        self.current_hour += 1
        self.entity.happiness = max(0.0, self.entity.happiness - 0.005)

        # 종료 조건
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
