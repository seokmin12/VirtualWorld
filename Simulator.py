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
        self.user = self.createUser()
        self.time = 0  # 현재 시각
        self.days = 0  # 생존 날짜
        self.week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']  # 요일
        self.mining_difficulty = 10000  # 채굴 난이도
        self.reward_per_block = 50  # 블록 당 보상 (SKM)

    def createUser(self):
        user = Entity('tommy')
        user.account = self.bank.createAccount(user.name)
        return user

    def mine(self):
        """블록체인 채굴 작업 수행"""
        if self.user.health <= 0:
            print("체력이 부족하여 채굴할 수 없습니다.")
            return

        # 현재 컨디션에 따른 수정계수 가져오기
        condition_stats = self.user.conditions[self.user.current_condition]
        mining_speed = condition_stats['mining_speed']
        health_consumption = condition_stats['health_consumption']

        # 채굴 난이도와 채굴 파워에 따른 채굴 시간 계산
        mining_time = self.mining_difficulty / (self.user.mining_power * mining_speed)
        self.time += mining_time

        # 체력 소모 계산 (시간당 20의 체력 소모를 기준으로)
        health_cost = (mining_time / 3600) * 20 * health_consumption
        self.user.health -= health_cost

        # 채굴 성공 확률 계산 (컨디션이 좋을수록 성공확률 증가)
        success_chance = min(0.9, mining_speed * 0.6)  # 최대 90% 성공확률
        
        if random.random() < success_chance:
            # 채굴 성공
            reward = self.reward_per_block
            self.user.account.deposit(reward)
            # print(f"채굴 성공! {reward} SKM을 획득했습니다.")
            # print(f"소요 시간: {mining_time:.2f}초")
            # print(f"소모된 체력: {health_cost:.2f}")
            # print(f"남은 체력: {self.user.health:.2f}")
            # print("-" * 50)
        # else:
        #     # 채굴 실패
        #     print("채굴 실패...")
        #     print(f"소요 시간: {mining_time:.2f}초")
        #     print(f"소모된 체력: {health_cost:.2f}")
        #     print(f"남은 체력: {self.user.health:.2f}")
        #     print("-" * 50)

        # 컨디션 변화 (채굴 후 컨디션이 나빠질 확률 증가)
        self._update_condition(mining_time)

    def _update_condition(self, elapsed_time):
        """채굴 시간에 따른 컨디션 변화"""
        condition_change_chance = min(0.8, elapsed_time / 3600)  # 최대 80% 확률로 컨디션 변화
        
        if random.random() < condition_change_chance:
            current_index = list(self.user.conditions.keys()).index(self.user.current_condition)
            if current_index < 2:  # high -> medium -> low
                self.user.current_condition = list(self.user.conditions.keys())[current_index + 1]
                # print(f"컨디션이 {self.user.current_condition}로 변화했습니다.")

    def rest(self):
        """휴식을 취하고 체력을 회복"""
        recovery = self.user.rest_recovery_rate
        self.user.health = min(100, self.user.health + recovery)
        
        # 휴식 후 컨디션이 좋아질 확률
        if random.random() < 0.4:  # 40% 확률로 컨디션 향상
            current_index = list(self.user.conditions.keys()).index(self.user.current_condition)
            if current_index > 0:  # low -> medium -> high
                self.user.current_condition = list(self.user.conditions.keys())[current_index - 1]
                # print(f"휴식으로 인해 컨디션이 {self.user.current_condition}로 향상되었습니다.")

        self.time += self.user.rest_time
        # print(f"휴식을 취했습니다. 체력이 {recovery}만큼 회복되었습니다.")
        # print(f"현재 체력: {self.user.health:.2f}")
        # print("-" * 50)

    def choose_action(self):
        """현재 상태를 고려하여 다음 행동을 선택"""
        if self.user.should_rest():
            self.rest()
        else:
            # 체력과 컨디션에 따른 채굴 확률 계산
            mining_chance = 0.7  # 기본 70% 확률로 채굴
            if self.user.current_condition == 'high':
                mining_chance = 0.9
            elif self.user.current_condition == 'low':
                mining_chance = 0.5
            
            if random.random() < mining_chance or self.user.health == 100:
                self.mine()
            else:
                self.rest()

    def run(self):
        while True:
            self.choose_action()
            if self.user.health == 0.00 or self.time >= 3600 * 24:
                print(f"{self.days + 1}일차 잔액 돈: {self.user.account.balance}")
                break
        self.days += 1
        self.time = 0
