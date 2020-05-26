
import numpy as np
import re
from csv import reader
import pandas
import collections
import nltk
import pickle

ds = []

with open("C:/Users/Kevin Spiceywhinner/Desktop/TestDataset/yahoo_answers_csv/train.csv", 'r', encoding='utf-8') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    for row in csv_reader:
        if row[0] == '7':
            ds.append(row)
