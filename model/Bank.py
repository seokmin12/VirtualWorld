import json
from model.Account import Account
from model.Transaction import Transaction
from typing import List
import random
import string


class Bank:
    def __init__(self, asset=1000000, interest_rate=2.5, payment_reserve_ratio=10):
        self.name: str = "International Bank"
        self.currency_unit: str = "SKM"
        self.accounts: List = []
        self.initAccount()
        self.asset: int = asset + sum([account.balance for account in self.accounts])  # 은행 자본 100만원 + 은행 고객들의 계좌 전액
        self.interest_rate = interest_rate
        self.can_loan_amount: int = int(self.asset * ((100 - payment_reserve_ratio) / 100))

    def initAccount(self) -> List:
        genesis_account = Account(account_number=0, name="Genesis Account", balance=0, previous_hash="0")
        self.accounts.append(genesis_account)

        with open('users.json', 'r') as f:
            users = json.load(f)
        for user in users:
            self.accounts.append(
                Account(
                    account_number=self.createRandomNum(),
                    name=user["name"],
                    balance=user["balance"],
                    previous_hash=self.getLatestAccount().hash
                )
            )

    def createRandomNum(self) -> str:
        while True:
            new_num = ''.join(random.choices(string.digits, k=10))
            if not any(account.account_number == new_num for account in self.accounts):
                return str(new_num)

    def loan(self, account_number: str, amount: int, private_key, public_key):
        if self.can_loan_amount > amount:
            for account in self.accounts:
                if account.getAccountNumber() == account_number:
                    transaction = Transaction(
                        self.name,
                        str(account_number),
                        amount
                    )
                    transaction.generateSignature(private_key)
                    if not transaction.verifySignature(public_key):
                        print("Invalid transaction signature.")
                        return
                    else:
                        account.transactions.append(transaction)
                        self.can_loan_amount -= amount
                        self.asset -= amount
                        account.balance += amount
                        print(f"The loan has been approved. Now your balance: {format(account.balance, ',')} {self.currency_unit}")
        else:
            print("The loan amount exceeds the loanable amount.")

    def transfer(self, sender_account_num, recipient_account_num, amount: int, private_key, public_key):
        sender_account = next((acc for acc in self.accounts if acc.getAccountNumber() == sender_account_num), None)
        recipient_account = next((acc for acc in self.accounts if acc.getAccountNumber() == recipient_account_num), None)

        if sender_account and recipient_account:
            transaction = Transaction(sender_account_num, recipient_account_num, amount)
            transaction.generateSignature(private_key)

            if not transaction.verifySignature(public_key):
                print("Invalid transaction signature.")
                return
            else:
                sender_account.transactions.append(transaction)
                sender_account.balance -= amount
                recipient_account.transactions.append(transaction)
                recipient_account.balance += amount
                print("Transfer successfully!")
        else:
            print("Didn't find that account, please check again.")

    def getLatestAccount(self) -> Account:
        return self.accounts[-1]
