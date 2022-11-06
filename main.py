#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from absl import flags
from dataclasses import dataclass, field, InitVar
from joblib import Memory
from tqdm import trange

from tensorflow import keras
from keras import optimizers
from keras import backend as K
#from tensorflow.keras import layers, models, regularizers, optimizers
from PIL import Image

memory = Memory(".cache")

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 1024, "Number of samples in a batch")
flags.DEFINE_integer("epochs", 5, "Number of epochs")
flags.DEFINE_float("lr", .1, "Learning rate for ADAM")
flags.DEFINE_integer("num_iters", 50000, "number of iterations for ADAM")
flags.DEFINE_string("optype", "style", "style or content extraction")
flags.DEFINE_string("impath", "starry_night.jpeg", "path to content or style image")


#make trainavar for gen content and style
class Data(tf.Module):
    def __init__(self, rng):

        self.cont = tf.Variable(rng.uniform(low=0.0,high=1.0,size= [1,224,224,3]), trainable = True, dtype=tf.float32)


@memory.cache()
def vgg_16():
    #IMPLEMENT VGG NETWORK
    #USE FEATURE SPACE OF 16 CONV AND 5 POOLING LAYERS OF 19 LAYER VGG
    #REPLACE MAX POOLING WITH AVERAGE POOLING
    model = VGG16(include_top=False, weights = "imagenet")

    print(model.summary())
    return model


def main():

    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    iters = FLAGS.num_iters
    optype = FLAGS.optype
    path = FLAGS.impath

    #random number gen
    np_rng = np.random.default_rng(31415)

    #initialize model

    optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)


    lossarr = np.zeros(iters, dtype=float)


if __name__ == "__main__":
    main()
