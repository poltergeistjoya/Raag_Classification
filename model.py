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


def conv_module(input, num_filters, activation, kern_reg, dropout, padding="same"):
    input = Conv2D(filters = num_filters, kernel_size = (3, 3), activation = activation, padding = padding, kernel_regularizer = kern_reg)(input)
    input = BatchNormalization(axis=-1)(input)

    return input

def double_conv_module(input, num_filters, activation, kern_reg, dropout, padding="same"):
    input = conv_module(input = input, filters = num_filters, kernel_size = (3, 3), activation = activation, padding = padding, kernel_regularizer = kern_reg)
    input = conv_module(input = input, filters = num_filters, kernel_size = (3, 3), activation = activation, padding = padding, kernel_regularizer = kern_reg)
    input = MaxPooling2D(pool_size = (2, 2))(input)
    input = Dropout(dropout)(input)

    return input

def rav_model(width, height, depth, classes):
    inputShape=(height, width, depth)
    weight_decay = 0.001

    inputs = Input(shape=inputShape)
    KR = None #l2(weight_decay)
    x = double_conv_module(inputs, 32, activation='relu', kern_reg=KR, dropout = 0.1, padding='same')
    x = double_conv_module(x, 64, activation='relu', kern_reg=KR, dropout = 0.2, padding='same')
    x = double_conv_module(x, 128, activation='relu', kern_reg=KR, dropout = 0.3, padding='same')
    x = double_conv_module(x, 128, activation='relu', kern_reg=KR, dropout = 0.4, padding='same')
   
    x = Flatten()(x)
    x = Dense(512, activation='relu',kernel_regularizer=None)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x, name="rav_net")
    return model

def simple_model(image_len, image_width, num_classes, lambda_val):
  inputShape = (image_len, image_width, 1)
  inputs = Input(shape=inputShape)

  output = Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(lambda_val))(inputs)
  output = BatchNormalization(name='batchnorm_1')(output)
  output = Activation('relu')(output)

  output = MaxPool2D(pool_size=(2,2))(output)
  output = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(lambda_val))(output)

  output = MaxPool2D(pool_size=(2,2))(output)
  output = Dropout(0.25)(output)
  output = Flatten()(output)
  output = Dense(128,activation='relu',kernel_regularizer=l2(lambda_val))(output)
  output = Dropout(0.5)(output)

  output = Dense(num_classes,activation='softmax')(output)

  model = Model(inputs, output, name="simple_model")

  return model

variable_learning_rate=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2)

model = simple_model(32, 32, 4, lambda_val = 1e-5)
model.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])

print(model.summary())

#history = model.fit(x_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[variable_learning_rate],validation_data=(x_val,y_val),verbose=1)
#score = model.evaluate(x_test,y_test,verbose=0)

#print('Test loss:',score[0])
#print('Test accuracy:',score[1])

# Could use this later
# https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/