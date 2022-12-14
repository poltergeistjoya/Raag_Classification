#import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import OneHotEncoder
import time

from os import path
from pydub import AudioSegment
from joblib import Memory

def output_2_rankings(logits, encoder, num_display = 5):
    indices = (-logits).argsort()[:len(logits)]
    for i, index in enumerate(indices):
        one_hot = np.zeros(10)
        one_hot[index] = 1
        print(encoder.inverse_transform(one_hot.reshape(1, -1))[0][0], logits[index], end = " || ")

        if i == num_display:
            break
