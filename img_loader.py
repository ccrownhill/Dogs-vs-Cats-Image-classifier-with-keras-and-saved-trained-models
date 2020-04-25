import os
import numpy as np
from PIL import Image as img
from random import shuffle

CLASS_NAMES = np.array(['cat', 'dog'])

def load_data(dir, width=224, height=224):
    lst = os.listdir(dir)
    shuffle(lst)
    x_train = np.zeros((len(lst), width, height, 3), dtype=np.float32)
    y_train = np.zeros((len(lst), 2,), dtype=np.float32)
    for i in range(len(lst)):
        path = dir + "/" + lst[i]
        x_train[i] = decode_img(path, width, height)
        y_train[i] = get_label(lst[i])
    return x_train, y_train


def decode_img(file, width, height):
    i = img.open(file)
    i = i.resize((width, height))
    return np.asarray(i, dtype="float32")

def get_label(filename):
    res = np.zeros((2,))
    if filename.split('.')[0] == CLASS_NAMES[0]:
        res[0] = 1.0
    elif filename.split('.')[0] == CLASS_NAMES[1]:
        res[1] = 1.0
    return res