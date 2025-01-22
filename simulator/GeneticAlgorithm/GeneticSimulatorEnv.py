import gym
from gym import spaces
import numpy as np
import random
from Simulator import Simulator
from model.Entity import Entity


class GeneticSimulatorEnv(gym.Env):
    def __init__(self, population_size=100):
        super().__init__()
        self.population_size = population_size
        self.population = [Entity(f"Agent_{i}") for i in range(population_size)]
        self.current_entity_idx = 0
        self.simulator = Simulator()
        # 현재 Entity의 계좌 생성
        self.simulator.user = self.population[self.current_entity_idx]
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
        entity = self.population[self.current_entity_idx]
        entity_state = entity.get_state_vector()
        return np.array(entity_state + [
            self.simulator.user.account.balance / 10000,  # 정규화된 잔액
            self.simulator.time / (24 * 3600)  # 정규화된 시간
        ], dtype=np.float32)

    def step(self, action):
        entity = self.population[self.current_entity_idx]
        
        # 행동에 따른 성공 확률 계산
        success_prob = random.random()
        action_success = False
        
        if action == 0:  # 채굴
            success_threshold = 0.5 + (entity.work_ethic * 0.3)
            action_success = success_prob < success_threshold
            if action_success:
                self.simulator.mine(entity)
                entity.total_mined += 1
            else:
                entity.health -= 0.1  # 실패 시 체력 감소
                
        elif action == 1:  # 휴식
            condition_values = {'low': 0, 'medium': 1, 'high': 2}
            condition_value = condition_values[entity.current_condition]
            recovery_bonus = 0.1 + (condition_value * 0.2)
            # recovery_bonus를 직접 전달하는 대신 entity의 rest_recovery_rate를 조정
            entity.rest_recovery_rate *= (1 + recovery_bonus)
            self.simulator.rest(entity)
            # 원래대로 복구
            entity.rest_recovery_rate /= (1 + recovery_bonus)
            action_success = True
            
        elif action == 2:  # 거래
            success_threshold = 0.4 + (entity.trading_skill * 0.4)
            action_success = success_prob < success_threshold
            if action_success:
                trade_profit = random.uniform(100, 1000) * (1 + entity.trading_skill)
                entity.account.balance += trade_profit
                entity.total_traded += 1
            else:
                loss = random.uniform(50, 500) * (1 - entity.trading_skill * 0.5)
                entity.account.balance = max(0, entity.account.balance - loss)
            
        # 상태 업데이트
        state = self._get_state()
        
        # 복합 보상 계산
        reward = self._calculate_reward(entity, action, action_success)
        
        done = (entity.health <= 0 or 
                self.simulator.time >= 3600 * 24 or
                entity.account.balance >= 1000000)  # 새로운 종료 조건
                
        if done:
            # 종합적인 적합도 계산
            fitness_score = (
                (entity.account.balance / 10000) * 0.4 +  # 잔액 가중치
                ((24 * 3600 - self.simulator.time) / (24 * 3600)) * 0.2 +  # 시간 효율성
                entity.health * 0.2 +  # 건강 상태
                (entity.total_mined / 100) * 0.1 +  # 채굴 실적
                (entity.total_traded / 100) * 0.1   # 거래 실적
            )
            
            entity.update_fitness(
                fitness_score,
                self.simulator.time
            )
            
        return state, reward, done, {"success": action_success}

    def _calculate_reward(self, entity, action, success):
        base_reward = 0
        
        # 기본 잔액 보상
        balance_reward = np.log1p(entity.account.balance) * 0.01
        
        # 행동별 보상
        action_reward = 0
        if action == 0:  # 채굴
            action_reward = 20 if success else -10
            action_reward *= (1 + entity.work_ethic)
        elif action == 1:  # 휴식
            action_reward = 5 if entity.health < 0.5 else -5
        elif action == 2:  # 거래
            action_reward = 30 if success else -20
            action_reward *= (1 + entity.trading_skill)
        
        # 건강 상태에 따른 페널티
        health_penalty = -50 if entity.health < 0.2 else 0
        
        # 시간 효율성 보상
        time_efficiency = 10 if self.simulator.time < 3600 * 12 else 0
        
        total_reward = (
            balance_reward + 
            action_reward + 
            health_penalty + 
            time_efficiency
        )
        
        # 위험 감수성에 따른 보상 조정
        risk_modifier = 1 + (entity.risk_tolerance - 0.5) * 0.4
        total_reward *= risk_modifier
        
        return total_reward

    def reset(self):
        self.simulator = Simulator()
        self.current_entity_idx = (self.current_entity_idx + 1) % self.population_size
        # 새로운 Entity의 계좌 생성
        self.simulator.user = self.population[self.current_entity_idx]
        self.simulator.user.account = self.simulator.bank.createAccount(self.simulator.user.name)
        return self._get_state()

    def evolve_population(self):
        """개선된 세대 진화"""
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        elite_size = self.population_size // 10  # 상위 10%
        new_population = self.population[:elite_size]
        
        # 토너먼트 선택
        def tournament_select(tournament_size=3):
            tournament = random.sample(self.population, tournament_size)
            return max(tournament, key=lambda x: x.fitness_score)
        
        # 나머지 90% 생성
        while len(new_population) < self.population_size:
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            if random.random() < 0.8:  # 80% 확률로 교배
                child = Entity.crossover(parent1, parent2)
            else:  # 20% 확률로 복제
                selected_parent = random.choice([parent1, parent2])
                child = Entity(f"Agent_{len(new_population)}")
                child.risk_tolerance = selected_parent.risk_tolerance
                child.work_ethic = selected_parent.work_ethic
                child.trading_skill = selected_parent.trading_skill
            
            # 적응적 돌연변이 확률
            mutation_rate = 0.1 * (1 - parent1.fitness_score / self.population[0].fitness_score)
            if random.random() < mutation_rate:
                child.mutate()
            
            new_population.append(child)
        
        self.population = new_population
