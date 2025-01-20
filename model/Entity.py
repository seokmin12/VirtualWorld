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
        self.actions = ['mine', 'rest']
        self.mining_power = 10  # 기본 채굴 파워
        self.rest_recovery_rate = 10  # 휴식 시 회복되는 체력
        self.rest_time = 3600  # 휴식 시간 (1시간)

    def should_rest(self) -> bool:
        """체력과 컨디션을 기반으로 휴식이 필요한지 판단"""
        if self.health < 30:  # 체력이 30% 미만이면 반드시 휴식
            print('should rest')
            return True
        elif self.current_condition == 'low' and self.health < 50:  # 컨디션이 나쁘고 체력이 50% 미만이면 휴식
            print('should rest')
            return True
        return False
