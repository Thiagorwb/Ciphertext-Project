import numpy as np
import csv
import string

def _read_csv(data_dir, fn):
    with open(data_dir+"/"+fn+".csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        return np.array(list(reader))

def _read_text(data_dir, fn):
    if data_dir == None:
        with open(fn+".txt", "r") as txtfile:
            return np.array(list(txtfile.readlines()[0][:-1]))

    with open(data_dir+"/"+fn+".txt", "r") as txtfile:
        return np.array(list(txtfile.readlines()[0][:-1]))


def to_index(text, alphabet):
    """ takes list of chars, returns numpy array of indices"""
    d = dict(zip(alphabet, range(len(alphabet))))
    return np.array([d[t] for t in text])

def to_text(code, alphabet):
    """ takes np.array of characters indices, returns list of chars"""
    d = dict(zip(range(len(alphabet)), alphabet))
    return [d[c] for c in code]
