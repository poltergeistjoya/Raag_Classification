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

import pandas as pd
import preprocessing
import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Generate and shuffle data 
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
df , enc = preprocessing.generate_dataset()
data = df.sample(frac=1).reset_index(drop=True)

# create optimizer, we use adam with a Learning rate of 1e-4
opt = Adam(learning_rate = 1e-4)

# Categorical Cross Entropy Loss Function
cce = tf.keras.losses.CategoricalCrossentropy

def train_step(batch):
    # Batch is a slice of the dataset, each sample consists of ('path to audio file', one_hot_class)
    # For each audio file in the batch, generate series of spectrograms that correspond to it. 
    data_x, data_y = preprocessing.create_batch(batch)

    with tf.GradientTape() as tape:
        prediction = model(data_x)
        loss_value = cce(data_y, prediction)
    
    gradients = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_value

epochs = 5

if False:
    for epoch in range(1, epochs+1):

        bar = tf.keras.utils.Progbar(len(train_ds)-1)
        losses = []

        # Iterate over the batches of the dataset.
        for i, batch in enumerate(iter(train_ds)):
            loss = train_step(batch[0], batch[1])
            losses.append(loss)
            bar.update(i, values=[("loss", loss)])  

        avg = np.mean(losses)
        print(f"Average loss for epoch {epoch}/{epochs}: {avg}")
        ckpt_manager.save(checkpoint_number=e)