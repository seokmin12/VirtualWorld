import random


class Apartment:
    def __init__(
            self,
            name: str,
            location: str,
            floor: int,
            condition: str,
            base_price: int,
            size: int = 32,
            rooms: int = 3,
    ):
        """
            아파트 속성 초기화
            :param name: 아파트 이름
            :param location: 위치 (예: "City Center")
            :param size: 크기 (평수, 제곱미터 등)
            :param rooms: 방 개수
            :param floor: 몇 층인지
            :param condition: 상태 (예: "New", "Old")
            :param base_price: 기본 가격
        """
        self.name = name
        self.location = location
        self.size = size
        self.rooms = rooms
        self.floor = floor
        self.condition = condition
        self.base_price = base_price
        self.price = base_price  # 초기 가격
        self.rent = round(base_price * 0.005, 2)  # 임대료는 기본 가격의 0.5%

    def fluctuate_price(self):
        """가격 변동 로직"""
        fluctuation = random.uniform(-0.1, 0.1)  # -10% ~ +10% 변동
        self.price = round(self.base_price * (1 + fluctuation), 2)
        self.rent = round(self.price * 0.005, 2)

    def __str__(self):
        return (
            f"Apartment: {self.name}, Location: {self.location}, Size: {self.size} sqm, Rooms: {self.rooms}, "
            f"Floor: {self.floor}, Condition: {self.condition}, Price: ${self.price:.2f}, Rent: ${self.rent:.2f}/month"
        )
