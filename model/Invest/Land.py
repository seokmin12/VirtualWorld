class Land:
    def __init__(
            self,
            name: str,
            area: int,
            zoning: str,  # 용도 지역 ex) 'Residential', 'Commercial', 'Agricultural'
            buildable: bool,  # 건축 가능 여부,
            base_price_per_sqm: int,
    ):
        """
        Land 클래스 초기화
        :param name: 땅 이름
        :param area: 면적 (제곱미터)
        :param zoning: 용도지역
        :param buildable: 건축 가능 여부
        :param base_price_per_sqm: 기본 평방미터당 가격
        """
        self.name = name
        self.area = area
        self.zoning = zoning
        self.buildable = buildable
        self.base_price_per_sqm = base_price_per_sqm
        self.owner = None
        self.price = self.calculate_price()

    def calculate_price(self):
        base_price = self.area * self.base_price_per_sqm
        return base_price

    def change_zoning(self, new_zoning: str):
        self.zoning = new_zoning
        print(f"{self.name} zoning  changed to {self.zoning}.")

    def change_owner(self, user):
        if self.owner == user:
            print(f"{user.name} already own {self.name}.")
            return
        self.owner = user
        print(f"{self.name}'s owner changed to {user.name}.")

    def __str__(self):
        return (
            f"Land: {self.name}, Area: {self.area} sqm, "
            f"Zoning: {self.zoning}, Buildable: {'Yes' if self.buildable else 'No'}, "
            f"Base Price per sqm: ${self.base_price_per_sqm:.2f}, Total Price: ${self.price:.2f}, "
        )
