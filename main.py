from model.Account import Account
from model.Bank import Bank
from model.Transaction import Transaction
from model.Invest.Stock import Stock
import rsa

# public_key, private_key = rsa.newkeys(2048)
#
# bank = Bank()
# user1 = bank.accounts[1].__dict__
# user2 = bank.accounts[2].__dict__
# bank.transfer(user1['account_number'], user2['account_number'], 10000, public_key=public_key, private_key=private_key)
# for tra in user1['transactions']:
#     print(tra.__dict__)
#
# bank.accounts[1].deposit(1000)
sam = Stock(
    name="Samsung",
    issued_shares=100000,
    init_price=65000,
    sigma=0.7,
    simulations=1
)

sam.run()
sam.getPriceForDay(1)
