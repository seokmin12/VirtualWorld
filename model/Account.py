from hashlib import sha256
from model.Transaction import Transaction
from typing import List


def apply_sha256(input):
    try:
        digest = sha256()
        digest.update(input.encode('utf-8'))
        hash_bytes = digest.digest()
        hex_string = ''.join(f'{b:02x}' for b in hash_bytes)
        return hex_string
    except Exception as e:
        print(e)
        raise RuntimeError(e)


class Account:
    def __init__(self, account_number: int, name: str, balance: int, previous_hash: str):
        self.account_number = account_number
        self.name = name
        self.balance = balance
        self.previous_hash = previous_hash
        self.hash: str = self.compute_hash()
        self.transactions: List = self.createGenesisTransaction()

    def deposit(self, amount: int) -> int:
        self.balance += amount
        print(f"{self.name} Deposited {amount} $. Current balance is: {self.balance}")
        return self.balance

    def withdraw(self, amount: int) -> int:
        if self.balance > amount:
            self.balance -= amount
            print(f"{self.name} Withdrew {amount} $. Current balance is: {self.balance}")
            return amount
        else:
            print("You don't have enough funds to withdraw.")
            return 0

    def compute_hash(self) -> str:
        calculatedHash = apply_sha256(
            str(self.account_number) + self.name + self.previous_hash
        )

        return calculatedHash

    def createGenesisTransaction(self):
        genesis_transaction = Transaction(0, 0, 0, "0")
        return [genesis_transaction]

    def getAccountNumber(self) -> str:
        return self.account_number

    def getAccountOwnerName(self) -> str:
        return self.name

    def getAccountBalance(self) -> int:
        return self.balance

    def setPreviousHash(self, previous_hash: str):
        self.previous_hash = previous_hash

    def setHash(self, hash: str):
        self.hash = hash
