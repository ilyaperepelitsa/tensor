from yahoo_finance import Share
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random
# import BeautifulSoup as bs

from iexfinance import get_historical_data
from datetime import datetime

start = datetime(2014, 2, 9)
end = datetime.now()

df = get_historical_data("AAPL", start=start, end=end, output_format='pandas')
df.head()
prices = list(df.close)

# def get_prices(share_symbol, start_date, end_date, cache_filename = "/Users/ilyaperepelitsa/Downloads/stock_prices.npy"):
#     try:
#         stock_prices = np.load(cache_filename)
#     except IOError:
#         share = Share(share_symbol)
#         stock_hist = share.get_historical(start_date, end_date)
#         stock_prices = [stock_price["Open"] for stock_price in stock_hist]
#         np.save(cache_filename, stok_prices)
#     return stock_price.astype(float)
#
# # import urllib2
# from bs4 import BeautifulSoup as bs
#
# def get_historical_data(name, number_of_days):
#     data = []
#     url = "https://finance.yahoo.com/quote/" + name + "/history/"
#     rows = bs(urllib2.urlopen(url).read()).findAll('table')[0].tbody.findAll('tr')
#
#     for each_row in rows:
#         divs = each_row.findAll('td')
#         if divs[1].span.text  != 'Dividend': #Ignore this row in the table
#             #I'm only interested in 'Open' price; For other values, play with divs[1 - 5]
#             data.append({'Date': divs[0].span.text, 'Open': float(divs[1].span.text.replace(',',''))})
#
#     return data[:number_of_days]
#
# get_historical_data("AAPL", 200)
def plot_prices(prices):
    plt.title("Opening stock prices")
    plt.xlabel("day")
    plt.ylabel("price ($)")
    plt.plot(prices)
    plt.savefig("prices.png")
    plt.show()

# prices = get_prices("AAPL", "1992-07-22", "2016-07-22")
plot_prices(prices)


class DecisionPolicy:
    def select_action(self, current_state):
        pass
    def update_q(self, state, action, reward, next_action):
        pass

class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state):
        action = random.choice(self.actions)
        return action

    def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist):
        budget = initial_budget
        num_stocks = initial_num_stocks
        share_value = 0
        transitions = list()
        for i in range(len(prices) - hist - 1):
            if i % 1000 == 0:
                print("progress {:.2f}%".format(float(100*i) / len(prices) - hist - 1)))
            current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))
            current_portfolio = budget + num_stocks * share_value
            action = policy.select_action(current_state, i)
            share_value = float(prices[i+hist])
            if action == "Buy" and budget >= share_value:
                budget -= share_value
                num_stocks += 1
            elif action == "Sell" and num_stocks > 0:
                budget += share_value
                num_stocks -= 1
            else:
                action = "Hold"
            new_portfolio = budget + num_stocks * share_value
            reward = new_portfolio - current_portfolio
            next_state = np.asmatrix(np.hstack((prices[i + 1 : i + hist + 1], budget, num_stocks)))
            transitions.append((current_state, action, reward, next_state))
            policy.update_q(current_state, action, reward, next_state)
        portfol
