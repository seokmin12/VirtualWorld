from model.Invest.Stock import Stock
from model.Transaction import Transaction
from typing import List
import json


class StockMarket:
    def __init__(self):
        self.stocks: List = []
        self.initStock()

    def initStock(self):
        with open('stocks.json', 'r') as f:
            stock_data = json.load(f)

        for data in stock_data:
            stock = Stock(
                name=data["name"],
                issued_shares=data["issued_shares"],
                init_price=data["init_price"],
                aver_profit_rate=data["aver_profit_rate"],
                sigma=data["sigma"]
            )
            stock.run()
            self.stocks.append(stock)

    def trade(self, account, stock_name: str, quantity: int, date: int, buy: bool, private_key, public_key):
        # 주식이 시장에 상장되어 있는지 확인
        stock_found = False
        for stock in self.stocks:
            if stock.name == stock_name:
                stock_found = True
                want_to_trade_stock = stock
                break

        if not stock_found:
            print(f"Stock {stock_name} not found!")
            return

        stock_price = want_to_trade_stock.getPriceForDay(date)
        total_cost = stock_price * quantity

        if buy:  # 주식 구매
            if account.balance >= total_cost:
                account.balance -= total_cost
                # 계좌 거래 내역 추가
                transaction = Transaction(
                    account.name,
                    f"Buy {stock_name}",
                    quantity
                )
                transaction.generateSignature(private_key)
                if not transaction.verifySignature(public_key):
                    print("Invalid transaction signature.")
                    return
                else:
                    account.transactions.append(transaction)

                    # 잔고에 입력할 주식 폼
                    stock_form = {
                        "buy_price": stock_price,
                        "quantity": quantity
                    }
                    # 잔고에 추가
                    if stock_name in account.assets["stocks"]:
                        # 평단가 계산
                        previous_total_buy_price = account.assets["stocks"][stock_name]["buy_price"] * \
                                                  account.assets["stocks"][stock_name]["quantity"]
                        new_aver_buy_price = (previous_total_buy_price + total_cost) / (
                                    account.assets["stocks"][stock_name]["quantity"] + quantity)

                        account.assets["stocks"][stock_name]["buy_price"] = new_aver_buy_price
                        account.assets["stocks"][stock_name]["quantity"] += quantity
                    else:
                        account.assets["stocks"][stock_name] = stock_form

                    print(f"Stock {stock_name} bought {quantity} at {stock_price} successfully!")
            else:
                print(f"{account.name} does not have enough money to buy {stock_name}.")
        else:  # 주식 판매
            if stock_name in account.assets["stocks"] and account.assets["stocks"][stock_name]["quantity"] >= quantity:
                # 계좌 거래 내역 추가
                transaction = Transaction(
                    f"Sell {stock_name}",
                    account.name,
                    quantity
                )
                transaction.generateSignature(private_key)
                if not transaction.verifySignature(public_key):
                    print("Invalid transaction signature.")
                    return
                else:
                    account.transactions.append(transaction)
                    # 평단가 계산
                    previous_total_buy_price = account.assets["stocks"][stock_name]["buy_price"] * \
                                              account.assets["stocks"][stock_name]["quantity"]
                    new_aver_buy_price = (previous_total_buy_price - total_cost) / (
                            account.assets["stocks"][stock_name]["quantity"] - quantity)

                    account.assets["stocks"][stock_name]["quantity"] -= quantity
                    # 계좌 돈 입금
                    account.balance += total_cost

                    # 새 평단가 적용
                    account.assets["stocks"][stock_name]["buy_price"] = new_aver_buy_price
                    profit = (stock_price - account.assets["stocks"][stock_name]["buy_price"]) * quantity
                    print(f"Stock {stock_name} selled {quantity} at {stock_price} successfully! Your profit: {profit}")

