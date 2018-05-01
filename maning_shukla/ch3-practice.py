import csv
import time


def read(filename, date_idx, date_parse, year, bucket = 7):

    days_in_year = 365

    freq = {}
    for period in range(0, int(days_in_year / bucket)):
        freq[period] = 0

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[date_idx] == "":
                continue
            t = time.strptime(row[date_idx], date_parse)
            if t.tm_year == year and t.tm_yday < (days_in_year - 1):
                freq[int(t.tm_yday / bucket)] += 1
    return freq

freq = read("/Users/ilyaperepelitsa/Downloads/311_Call_Center_Tracking_Data__Archived_.csv", 0, "%m/%d/%Y", 2014)
freq
