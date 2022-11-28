import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Add, Dense, Activation, BatchNormalization, Lambda, ReLU, PReLU
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Input, concatenate, ZeroPadding2D, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# create optimizer, we use adam with a Learning rate of 1e-4
opt = keras.optimizers.Adam(learning_rate = 1e-4)

# Categorical Cross Entropy Loss Function
cce = tf.keras.losses.CategoricalCrossentropy


def train_step(batch):
    # Batch is a slice of the dataset, each sample consists of ('path to audio file', one_hot_class)
    # For each audio file in the batch, generate series of spectrograms that correspond to it. 
    for data_sample in batch:
        # Function that takes in file path and returns series of spectrograms
        


    with tf.GradientTape() as tape:
        prediction = model(noised_image, timestep_values)
        
        loss_value = cce(noise, prediction)
    
    gradients = tape.gradient(loss_value, unet_con.trainable_variables)
    opt.apply_gradients(zip(gradients, unet_con.trainable_variables))

    return loss_value

epochs = 5

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