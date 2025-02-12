import gym
import numpy as np
import random
from module.Bank import Bank
from module.Entity import Entity


class MultipleEntityEnv(gym.Env):
    def __init__(self):
        self.bank = Bank()
        self.entities = [Entity(f"entity{i}", self.bank) for i in range(2)]
        self.action_space = gym.spaces.Discrete(5)  # mine, rest, leisure, religious_activity, trade
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

    def step(self, action_list: list):
        states = []
        rewards = []
        dones = []
        infos = []
        for entity, action in zip(self.entities, action_list):
            if entity.terminated:
                states.append(entity.get_state())
                rewards.append(0.0)
                dones.append(True)
                infos.append({
                    "entity": entity.name,
                    "current_day": entity.current_day,
                    "work_hours": entity.work_hours,
                    "health": entity.health,
                    "balance": entity.account.balance,
                    "happiness": entity.happiness
                })
                continue

            reward: float = 0.0
            if entity.current_hour >= 24:
                if entity.work_hours > 12:
                    entity.health -= 10
                entity.current_day += 1
                if entity.current_day % 365 == 0:
                    entity.age += 1
                entity.current_hour = 0
                entity.work_hours = 0

            if action == 0:
                reward += entity.mine()
            elif action == 1:
                reward += entity.rest()
            elif action == 2:
                reward += entity.leisure()
            elif action == 3:
                reward += entity.religious_activity()
            elif action == 4:
                trade_amount = round(random.uniform(0.0, entity.account.balance), 0)
                other_entity = random.choice([e for e in self.entities if e != entity])
                reward += entity.trade(other_entity, trade_amount)
            else:
                reward -= 1.0

            survival_reward = 0.5 * (entity.health / 100.0) + 0.5 * (entity.happiness)
            reward += survival_reward

            entity.current_hour += 1
            entity.happiness = max(0.0, entity.happiness - 0.005)

            state = entity.get_state()

            done = False
            if entity.health <= 0 or entity.age >= entity.lifespan or entity.happiness < 0.05:
                done = True
                entity.terminated = True

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append({
                "entity": entity.name,
                "current_day": entity.current_day,
                "work_hours": entity.work_hours,
                "health": entity.health,
                "balance": entity.account.balance,
                "happiness": entity.happiness
            })

        return states, rewards, dones, infos

    def reset(self):
        self.entities = [Entity(f"entity{i}", self.bank) for i in range(2)]
        states = [entity.get_state() for entity in self.entities]
        return states, {}

