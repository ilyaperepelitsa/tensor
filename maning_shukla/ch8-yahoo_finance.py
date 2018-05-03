from yahoo_finance import Share
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random


# def get_prices(share_symbol, start_date, end_date, cache_filename = "/Users/ilyaperepelitsa/Downloads/stock_prices.npy"):
#     try:
#         stock_prices = np.load(cache_filename)
#     except IOError:
#         share = Share(share_symbol)
#         stock_hist = share.get_historical(start_date, end_date)
#         stock_prices = [stock_price["Open"] for stock_price in stock_hist]
#         np.save(cache_filename, stok_prices)
#     return stock_price.astype(float)

import urllib2
from BeautifulSoup import BeautifulSoup as bs

def get_historical_data(name, number_of_days):
    data = []
    url = "https://finance.yahoo.com/quote/" + name + "/history/"
    rows = bs(urllib2.urlopen(url).read()).findAll('table')[0].tbody.findAll('tr')

    for each_row in rows:
        divs = each_row.findAll('td')
        if divs[1].span.text  != 'Dividend': #Ignore this row in the table
            #I'm only interested in 'Open' price; For other values, play with divs[1 - 5]
            data.append({'Date': divs[0].span.text, 'Open': float(divs[1].span.text.replace(',',''))})

    return data[:number_of_days]


def plot_prices(prices):
    plt.title("Opening stock prices")
    plt.xlabel("day")
    plt.ylabel("price ($)")
    plt.plot(prices)
    plt.savefig("prices.png")
    plt.show()

prices = get_prices("AAPL", "1992-07-22", "2016-07-22")

float("pew")

def sum_nums(args):
    sum = 0
    for i in args:
        try:
            i = float(i)
            sum += i
        except ValueError:
            pass
    return sum
