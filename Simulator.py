import random
import time
from model.Bank import Bank
from model.StockMarket import StockMarket
from model.Entity import Entity
from typing import List


class Simulator:
    def __init__(self):
        self.bank = Bank()
        self.stock_market = StockMarket()
        self.user = None
        self.time = 0           # Simulation time in seconds
        self.days = 0
        self.week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.mining_difficulty = 10000  # Mining difficulty
        self.reward_per_block = 50      # Reward per successful mining block

    def run(self, entity: Entity) -> None:
        """
        Run a single day of simulation.
        The simulation stops when the entity's health is depleted or a day is complete.
        """
        while True:
            if entity.health <= 0 or self.time >= 3600 * 24:
                if entity.account:
                    print(f"Day {self.days + 1}: Balance: {entity.account.balance:.2f}")
                break
        self.days += 1
        self.time = 0
