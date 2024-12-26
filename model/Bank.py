import json
from model.Account import Account
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

    def createRandomNum(self) -> int:
        while True:
            new_num = int(''.join(random.choices(string.digits, k=10)))
            if not any(account.account_number == new_num for account in self.accounts):
                return new_num

    def loan(self, account_number: str, amount: int):
        if self.can_loan_amount > amount:
            self.can_loan_amount -= amount
            self.asset -= amount
            for account in self.accounts:
                if account.getAccountNumber() == account_number:
                    account.balance += amount
                    print(f"The loan has been approved. Now your balance: {format(account.balance, ',')} {self.currency_unit}")
        else:
            print("The loan amount exceeds the loanable amount.")

    def getLatestAccount(self) -> Account:
        return self.accounts[-1]
