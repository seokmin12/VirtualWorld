from model.Account import Account
from model.Bank import Bank
from model.Transaction import Transaction
import rsa

public_key, private_key = rsa.newkeys(2048)

bank = Bank()
user1 = bank.accounts[1].__dict__['account_number']
bank.loan(user1, 10000, public_key=public_key, private_key=private_key)
