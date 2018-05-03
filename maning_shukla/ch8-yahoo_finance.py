from yahoo_finance import Share
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random


def get_prices(share_symbol, start_date, end_date, cache_filename = "stock_prices.npy"):
    try:
        stock_prices = np.load(cache_filename)
    except IOError:
        share = Share(share_symbol)
        stock_hist = share.get_historical(start_date, end_date)
        stock_prices = [stock_price["Open"] for stock_price in stock_hist]
        np.save(cache_filename, stok_prices)
    return stock_price.astype(float)

def plot_prc
