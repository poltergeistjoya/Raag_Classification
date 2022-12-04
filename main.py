#trains but there are some confusions and concerns
#simple model needs input shape (None, width, len, 1) why is there a none?
#^above is cause of concern for pipeline
#set epochs to 5, but when training it gives 17 ?
#loss is behaving very arbitrarily

#add validation step to train step?


import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from absl import flags

import pandas as pd
import preprocessing
import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 0.1, "Learning Rate")
flags.DEFINE_integer("epochs", 5, "Number of epochs")
flags.DEFINE_integer("batch_size", 4, "Num Audio files in a batch")
flags.DEFINE_string("ds_path", "./recordings", "Path to dataset")
flags.DEFINE_integer("rand", 31415, "random seed")


def train_step(model, batch, loss, opt):
    # Batch is a slice of the dataset, each sample consists of ('path to audio file', one_hot_class)
    # For each audio file in the batch, generate series of spectrograms that correspond to it.
    data_x, data_y = preprocessing.create_batch(batch)

    with tf.GradientTape() as tape:
        for i in range(0, len(data_x)):
            prediction = model(np.expand_dims(np.expand_dims(data_x[i], axis =2), axis=0))
            loss_value = loss(np.expand_dims(data_y[i], axis =1).T, prediction)
            #print(loss_value, type(loss_value))

    gradients = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_value

def main():
    #Parse flags
    FLAGS(sys.argv)
    LAMBDA = FLAGS.lr
    EPOCHS = FLAGS.epochs
    BATCH_SIZE = FLAGS.batch_size
    DS_PATH = FLAGS.ds_path
    RAND = FLAGS.rand

    # Generate and shuffle data
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    df , enc = preprocessing.generate_dataset(DS_PATH)
    df = df.sample(frac=1, random_state=RAND).reset_index(drop=True)
    # Should split off a small test set here later on

    # List of dataframes, where each entry is a single batch
    list_df = [df[i:i+BATCH_SIZE] for i in range(0,df.shape[0],BATCH_SIZE)]

    # Split the list into train, test, val
    list_train, list_val, list_test = np.split(list_df, [int(len(list_df)*0.8), int(len(list_df)*0.9)])
    #print(len(list_df), len(list_train), len(list_val), len(list_test))

    #create 1 batch to get correct IMAGE_LEN, IMAGE_WIDTH
    temp_x, temp_y = preprocessing.create_batch(list_train[0])
    #print("temp_x", len(temp_x), temp_x, "temp_y", len(temp_y), temp_y)
    #print("temp_x data shape", temp_x[0].shape, "temp_y data shape", temp_y[0].shape)
    IMAGE_LEN = temp_x[0].shape[0]
    IMAGE_WIDTH = temp_x[0].shape[1]
    NUM_RAGAS = temp_y[0].shape[0]


    # Create model
    model = models.simple_model(IMAGE_LEN, IMAGE_WIDTH, NUM_RAGAS, LAMBDA)

    # create optimizer, we use adam with a Learning rate of 1e-4
    opt = Adam(learning_rate = 1e-4)

    # Categorical Cross Entropy Loss Function
    cce = tf.keras.losses.CategoricalCrossentropy()

    for epoch in range(1, EPOCHS+1):
        bar = tf.keras.utils.Progbar(len(list_df)-1)
        losses = []

        # Iterate over the batches of the dataset.
        for i, batch in enumerate(iter(list_df)):
            loss = train_step(model, batch, cce,opt)
            losses.append(loss)
            bar.update(i, values=[("loss", loss)])

        avg = np.mean(losses)
        print(f"Average loss for epoch {epoch}/{EPOCHS}: {avg}")
        # ckpt_manager.save(checkpoint_number=e)

if __name__ == "__main__":
    main()
