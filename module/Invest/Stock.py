import numpy as np
import random
import matplotlib.pyplot as plt


class Stock:
    def __init__(
            self,
            name: str,
            issued_shares: int,
            init_price=0,
            aver_profit_rate=0.1,
            sigma=0.2,
            period=1,
            simulations=1
    ):
        self.paths = None
        self.name = name
        self.issued_shares = issued_shares  # 발행 주식 수
        self.init_price = init_price  # 초기 주식 가격
        self.aver_profit_rate = aver_profit_rate  # 평균 수익률
        self.sigma = sigma  # 변동성
        self.period = period  # 시뮬레이션 기간 (1년)
        self.dt = 1 / 252  # 일일 시간 간격 (주식 시장은 보통 252 거래일 기준)
        self.N = int(self.period / self.dt)  # 시간 단계 수
        self.simulations = simulations  # 시뮬레이션 수

    def run(self):
        np.random.seed(42)
        self.paths = np.zeros((self.simulations, self.N))
        self.paths[:, 0] = self.init_price

        for i in range(1, self.N):
            Z = np.random.normal(0, 1, self.simulations)  # 정규분포를 따르는 랜덤 변수
            self.paths[:, i] = (self.paths[:, i - 1] * np.exp(
                (self.aver_profit_rate - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)).astype(int)

    def displayChart(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.paths.T, color="blue", alpha=0.6)
        plt.title(f"{self.name} Stock Price")
        plt.xlabel('Time (Days)')
        plt.ylabel('Stock Price')
        plt.show()

    def getPriceForDay(self, day) -> int:
        price = self.paths.T[1 - day]
        return int(price[0])
