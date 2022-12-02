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
IMAGE_LEN = 5
IMAGE_WIDTH = 5
NUM_RAGAS = 3
LAMBDA = 0.1
EPOCHS = 5
BATCH_SIZE = 4 # Num Audio files in a batch 

# Generate and shuffle data 
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
df , enc = preprocessing.generate_dataset()
df = df.sample(frac=1).reset_index(drop=True)
# Should split off a small test set here later on

# List of dataframes, where each entry is a single batch
list_df = [df[i:i+BATCH_SIZE] for i in range(0,df.shape[0],BATCH_SIZE)]

# Split the list into train, test, val
#

# Create model
model = model.simple_model(IMAGE_LEN, IMAGE_WIDTH, NUM_RAGAS, LAMBDA)

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

def main():
    for epoch in range(1, EPOCHS+1):
        bar = tf.keras.utils.Progbar(len(list_df)-1)
        losses = []

        # Iterate over the batches of the dataset.
        for i, batch in enumerate(iter(list_df)):
            loss = train_step(batch)
            losses.append(loss)
            bar.update(i, values=[("loss", loss)])  

        avg = np.mean(losses)
        print(f"Average loss for epoch {epoch}/{EPOCHS}: {avg}")
        # ckpt_manager.save(checkpoint_number=e)

if __name__ == "__main__":
    print(list_df[0])
    main()