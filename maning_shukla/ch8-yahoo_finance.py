from yahoo_finance import Share
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random


def get_prices(share_symbol, start_date, end_date, cache_filename = "stock_prices.npy"):
    try:
        
