from model.Account import Account
from model.Bank import Bank
from model.Transaction import Transaction
from model.Invest.Stock import Stock
from model.StockMarket import StockMarket
from model.RealEstateMarket import RealEstateMarket
import rsa
from Simulator import Simulator
import time

# public_key, private_key = rsa.newkeys(2048)

# bank = Bank()
# user1 = bank.accounts[1]
# user2 = bank.accounts[2]

# stock_market = StockMarket()
# stock_market.trade(account=user1, stock_name="Tesla", quantity=100, date=1, private_key=private_key,
#                    public_key=public_key, buy=True)
# stock_market.trade(account=user1, stock_name="Tesla", quantity=123, date=5, private_key=private_key,
#                    public_key=public_key, buy=True)
# stock_market.trade(account=user1, stock_name="Tesla", quantity=1353, date=7, private_key=private_key,
#                    public_key=public_key, buy=True)
# stock_market.trade(account=user1, stock_name="Tesla", quantity=200, date=25, private_key=private_key,
#                    public_key=public_key, buy=False)
# print(user1.assets)

# real_estate_market = RealEstateMarket()
# real_estate_market.buy_land(user1, 'namoe-dong')
simulator = Simulator()
for _ in range(365):
    simulator.run()
    print(simulator.user.health)

