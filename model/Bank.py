import json
from model.Account import Account
from model.Transaction import Transaction
from typing import List
import random
import string


class Bank:
    def __init__(self, asset=1000000, interest_rate=2.5, payment_reserve_ratio=10):
        self.name: str = "International Bank"
        self.currency_unit: str = "SKM"  # 통화 단위
        self.accounts: List = []
        self.initAccount()  # 초기 계좌 생성
        self.asset: int = asset + sum([account.balance for account in self.accounts])  # 은행 자본 100만원 + 은행 고객들의 계좌 전액
        self.interest_rate = interest_rate  # 금리
        self.can_loan_amount: int = int(self.asset * ((100 - payment_reserve_ratio) / 100))  # 지급 준비율: 10%, 대출 가능 금액: 은행 자본의 90%

    def initAccount(self) -> List:
        # 제네시스 계좌(초기 계좌) 생성
        genesis_account = Account(account_number=0, name="Genesis Account", balance=0, previous_hash="0")
        self.accounts.append(genesis_account)

    def createAccount(self, name: str):
        account = Account(
                account_number=self.createRandomNum(),
                name=name,
                balance=0,
                previous_hash=self.getLatestAccount().hash
        )
        self.accounts.append(account)
        print(f"{name}'s account created successfully!")
        return account

    # 계좌 번호 10자리 랜덤 생성
    def createRandomNum(self) -> str:
        while True:
            new_num = ''.join(random.choices(string.digits, k=10))
            # 계좌 번호 중복 확인
            if not any(account.account_number == new_num for account in self.accounts):
                return str(new_num)

    def loan(self, account_number: str, amount: int, private_key, public_key):
        if self.can_loan_amount >= amount:  # 대출 가능 금액 >= 대출액
            for account in self.accounts:
                if account.getAccountNumber() == account_number:  # 계좌 번호 확인
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
                        # 대출 실행
                        account.transactions.append(transaction)
                        self.can_loan_amount -= amount  # 대출 가능 금액에서 대출액 차감
                        self.asset -= amount  # 은행 자본에서 대출액 차감
                        account.balance += amount  # 고객의 계좌에 돈 입금
                        print(f"The loan has been approved. Now your balance: {format(account.balance, ',')} {self.currency_unit}")
        else:
            print("The loan amount exceeds the loanable amount.")

    def transfer(self, sender_account_num, recipient_account_num, amount: int, private_key, public_key):
        # 계좌 번호 확인
        sender_account = next((acc for acc in self.accounts if acc.getAccountNumber() == sender_account_num), None)
        recipient_account = next((acc for acc in self.accounts if acc.getAccountNumber() == recipient_account_num), None)

        if sender_account and recipient_account:
            if sender_account.balance >= amount:  # 송금자의 계좌 잔액 >= 송금액
                transaction = Transaction(sender_account_num, recipient_account_num, amount)
                transaction.generateSignature(private_key)

                if not transaction.verifySignature(public_key):
                    print("Invalid transaction signature.")
                    return
                else:
                    # 송금자 잔액 차감
                    sender_account.transactions.append(transaction)
                    sender_account.balance -= amount
                    # 수신자 잔액 증가
                    recipient_account.transactions.append(transaction)
                    recipient_account.balance += amount
                    print("Transfer successfully!")
            else:
                print("Sender doesn't have enough money.")
        else:
            print("Didn't find that account, please check again.")

    def getLatestAccount(self) -> Account:
        return self.accounts[-1]
