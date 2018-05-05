import csv
import numpy as np
import matplotlib.pyplot as plt

def load_series(filename, series_idx = 1):
    try:
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile)

            data = [float(row[series_idx]) for row in csvreader if len(row) > 0]

            normalized_data = (data - np.mean(data)) / np.std(data)

        return normalized_data
    except IOError:
        return None

def split_data(data, percent_train = 0.8):
    num_rows = len(data) * percent_train
    return data[:num_rows], 
