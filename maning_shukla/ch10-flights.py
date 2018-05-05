import csv
import numpy as np
import matplotlib.pyplot as plt

def load_series(filename, series_idx = 1):
    try:
        with open(filename) as csvf
