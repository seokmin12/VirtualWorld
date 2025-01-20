import random

class Entity:
    def __init__(self, name: str):
        self.name = name
        self.health = 100.00  # 체력
        self.age = 0  # 나이
        self.lifespan = 100  # 수명 (기본값 100년)
        self.conditions = {
            'high': {'mining_speed': 1.5, 'health_consumption': 0.7},  # 컨디션이 좋을 때: 채굴 속도 증가, 체력 소모 감소
            'medium': {'mining_speed': 1.0, 'health_consumption': 1.0},  # 보통 컨디션: 기본 수치
            'low': {'mining_speed': 0.7, 'health_consumption': 1.3}  # 나쁜 컨디션: 채굴 속도 감소, 체력 소모 증가
        }
        self.current_condition = 'medium'  # 현재 컨디션
        self.actions = ['mine', 'rest', 'trade']  # 거래 액션 추가
        self.mining_power = 10  # 기본 채굴 파워
        self.rest_recovery_rate = 10  # 휴식 시 회복되는 체력
        self.rest_time = 3600  # 휴식 시간 (1시간)
        
        # 강화학습을 위한 추가 속성
        self.fitness_score = 0  # 적합도 점수
        self.total_mined = 0    # 총 채굴량
        self.total_traded = 0   # 총 거래량
        self.survival_time = 0  # 생존 시간
        
        # 유전 알고리즘을 위한 특성
        self.risk_tolerance = random.random()      # 위험 감수성 (0-1)
        self.work_ethic = random.random()         # 작업 윤리 (0-1)
        self.trading_skill = random.random()      # 거래 기술 (0-1)

    def should_rest(self) -> bool:
        """체력과 컨디션을 기반으로 휴식이 필요한지 판단"""
        if self.health < 30:  # 체력이 30% 미만이면 반드시 휴식
            print('should rest')
            return True
        elif self.current_condition == 'low' and self.health < 50:  # 컨디션이 나쁘고 체력이 50% 미만이면 휴식
            print('should rest')
            return True
        return False

    def get_state_vector(self):
        """현재 상태를 벡터로 반환"""
        condition_map = {'low': 0, 'medium': 1, 'high': 2}
        return [
            self.health / 100.0,                    # 정규화된 체력
            condition_map[self.current_condition] / 2.0,  # 정규화된 컨디션
            self.mining_power / 100.0,              # 정규화된 채굴력
            self.risk_tolerance,
            self.work_ethic,
            self.trading_skill
        ]
        
    def update_fitness(self, balance, survival_time):
        """적합도 점수 업데이트"""
        self.fitness_score = (
            balance * 0.4 +                  # 잔액 가중치
            self.total_mined * 0.2 +        # 채굴량 가중치
            self.total_traded * 0.2 +       # 거래량 가중치
            survival_time * 0.2             # 생존 시간 가중치
        )
        return self.fitness_score
        
    def mutate(self, mutation_rate=0.1):
        """특성 돌연변이"""
        if random.random() < mutation_rate:
            self.risk_tolerance = max(0, min(1, self.risk_tolerance + random.uniform(-0.1, 0.1)))
        if random.random() < mutation_rate:
            self.work_ethic = max(0, min(1, self.work_ethic + random.uniform(-0.1, 0.1)))
        if random.random() < mutation_rate:
            self.trading_skill = max(0, min(1, self.trading_skill + random.uniform(-0.1, 0.1)))
            
    @classmethod
    def crossover(cls, parent1, parent2):
        """두 개체의 교배"""
        child = cls(f"Child_{random.randint(1000, 9999)}")
        # 특성 혼합
        child.risk_tolerance = (parent1.risk_tolerance + parent2.risk_tolerance) / 2
        child.work_ethic = (parent1.work_ethic + parent2.work_ethic) / 2
        child.trading_skill = (parent1.trading_skill + parent2.trading_skill) / 2
        return child
