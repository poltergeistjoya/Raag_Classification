#!/usr/bin/env python3
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import time

from absl import flags
from joblib import Memory

from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Add, Dense, Activation, BatchNormalization, Lambda, ReLU, PReLU
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Input, concatenate, ZeroPadding2D, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

memory = Memory(".cache")

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 1024, "Number of samples in a batch")
flags.DEFINE_integer("epochs", 20, "Number of epochs")
flags.DEFINE_float("lr", .1, "Learning rate for ADAM")
flags.DEFINE_integer("num_iters", 50000, "number of iterations for ADAM")
flags.DEFINE_string("ds_path", "./recordings", "path to dataset")
flags.DEFINE_integer("seed", 31415, "random seed for reproducible results")

def to_decibles(signal, method = 'librosa'):
    # Perform short time Fourier Transformation of signal and take absolute value of results
    if method == 'librosa':
        stft = np.abs(librosa.stft(signal))
    else:
        frame_length = 4096
        frame_step = 1024

        stft = tf.signal.stft(
            signal,
            frame_length,
            frame_step,
            pad_end=False
        )
    # Convert to dB
    D = librosa.amplitude_to_db(stft, ref = np.max) # Set reference value to the maximum value of stft.

    return D # Return converted audio signal

# Function to plot the converted audio signal for checking
def plot_spec(D, sr, raag = "raag"):
    fig, ax = plt.subplots(figsize = (30,10))
    spec = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title = 'Spectrogram of ' + raag)
    fig.colorbar(spec)
    plt.show()
    plt.savefig("D")

    return spec

    '''
    Iterate through directory and collect the relative path of each recording.
    Parameters: x
        - dataset_path: the absolute path to the directory containing all of the data. Each subdirectory inside of this should
        contain all of the recordings for a specific raga, and the name of the subdirectory should be the common name of the raga
        in question.
    Returns:
        - df: dataframe containing the paths to each audio file, the name of the raga they correspond to, and the one-hot encoded version
        of the raga names
        - enc: the OneHotEncoder using to encode the ragas (so we can invert the process later)
    '''
def generate_dataset(dataset_path):
    raga_dict = dict()
    raga_directories = next(os.walk(dataset_path))[1]

    for raga_directory in raga_directories:
        recordings_path = os.path.join(dataset_path, raga_directory)
        filenames = [os.path.join(recordings_path, x) for x in next(os.walk(recordings_path), (None, None, []))[2]]
        raga_dict[raga_directory] = filenames

    # Generate master list where each entry is [path to audio file, name of Raag]
    master_list = []

    for raga in raga_dict.keys():
        expanded_list = [[x, raga] for x in raga_dict[raga]]
        master_list += expanded_list

    df = pd.DataFrame (master_list, columns = ['File path', 'Raga'])
    ragas = df.Raga.values

    enc = OneHotEncoder(handle_unknown='ignore')
    ragas_onehot = enc.fit_transform(ragas.reshape(-1,1)).toarray()

    # https://stackoverflow.com/questions/35565376/insert-list-of-lists-into-single-column-of-pandas-df
    df['Raga One-Hot'] = pd.Series(list(ragas_onehot))

    # print(ragas)
    # print(enc.inverse_transform(ragas))

    #display(df.to_string())

    return df, enc

@memory.cache()
def create_batch(dataset):
    '''
    To conserve memory, the dataset will only consist of a list of filenames and the raga they correspond too.
    At runtime, this function will be called periodically to retrieve the next batch of audio data.
    Process:
        - Slice the audio file into 30s chunks
        - Take the STFT of each chunk and pair it with the corresponding class
        - Return list spectrogram, one hot encoded pairs
    '''
    batch_x = []
    batch_y = []

    dataset = dataset.reset_index()

    for index, audiofile in dataset.iterrows():
        t = time.time()
        offset = 0.0
        duration = 30.0
        target_sr = 8000
        file_length = librosa.get_duration(filename = audiofile['File path'])

        while offset + duration < file_length:
            # Load the audio in to a np array
            y, sr = librosa.load(audiofile['File path'], sr=None, offset = offset, duration = 30.0)

            # Resample the audio to a consistent sampling rate, pad/truncate as needed
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            # Padding when
            if len(y) != duration * target_sr:
                y  = librosa.util.fix_length(y, size = duration * target_sr)
                print("padded")

            # Any sort of data augmentation (optional - can add in later)

            # Take STFT
            spec = to_decibles(y)

            # Add data to batch
            batch_x.append(spec)
            batch_y.append(audiofile['Raga One-Hot'])

            # Increment Offset
            offset += duration

        print(f'Elapsed: {time.time() - t}')

    #img = plot_spec(batch_x[10], target_sr)

    return batch_x, batch_y, #img

def conv_module(input, num_filters, activation, kern_reg, dropout, padding="same"):
    input = Conv2D(filters = num_filters, kernel_size = (3, 3), activation = activation, padding = padding, kernel_regularizer = kern_reg)(input)
    input = BatchNormalization(axis=-1)(input)

    return input


@memory.cache()
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
  print(model.summary())

  return model


def main():

    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    iters = FLAGS.num_iters
    ds_path = FLAGS.ds_path
    seed = FLAGS.seed

    #dataset with file paths and one hot encoded labels
    data, encoder = generate_dataset(ds_path)

    #train 80%, validate 10%, test 10% dfs
    train_df, rest_df = train_test_split(data, test_size= 0.2, random_state = seed)
    val_df, test_df = train_test_split(rest_df, test_size = 0.5, random_state = seed)

    #print(train_df.shape, val_df.shape, test_df.shape, val_df, test_df)

    #x, y, img = create_batch(val_df)
    test_x, test_y = create_batch(test_df)
    val_x,val_y = create_batch(val_df)
    train_x, train_y = create_batch(train_df)

    #print(img.shape)

    #print(x[10], x[10].shape)
    #print(x[10].shape)
    #plt.imshow(x[10])
    #plt.show()
    #plt.savefig('specplot.png')

    #print(x[11].shape)
    #plt.imshow(x[11])
    #plt.show()
    #plt.savefig('specplot2.png')

    #doesn't look like this is used yet
    variable_learning_rate=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2)


    #initialize model
    model = simple_model(32, 32, 4, lambda_val = 1e-5)
    model.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
    history = model.fit(train_x, train_y,
            steps_per_epoch = (train_x.shape[0])//batch_size,
            epochs = epochs,
            batch_size=batch_size,
            validation_data=(val_x, val_y),
            verbose = 1)

    test_lost, test_acc = model.evaluate(test_x, test_y, verbose = 2)

    #PLOTTING
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 100])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./epochaccuracy.pdf')




if __name__ == "__main__":
    main()
