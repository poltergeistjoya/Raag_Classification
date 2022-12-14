import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
from postprocessing import output_2_rankings

from tensorflow.keras.optimizers import Adam

from absl import flags

import models
import preprocessing

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 0.1, "Learning Rate")
flags.DEFINE_integer("epochs", 9, "Number of epochs")
flags.DEFINE_integer("batch_size", 6, "Num Audio files in a batch")
flags.DEFINE_string("ds_path", "./15raag/raga-data/10raags/", "Path to dataset")
flags.DEFINE_integer("rand", 31415, "random seed")

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
    list_train, list_test = np.split(list_df, [int(len(list_df)*0.8)])
    #print(len(list_df), len(list_train), len(list_val), len(list_test))

    #create 1 batch to get correct IMAGE_LEN, IMAGE_WIDTH
    temp_x, temp_y = preprocessing.create_batch_2(pd.DataFrame(list_train[0], columns = ['File path', 'Raga', 'Raga One-Hot']))
    IMAGE_LEN = temp_x[0].shape[0]
    IMAGE_WIDTH = temp_x[0].shape[1]
    NUM_RAGAS = temp_y[0].shape[0]

    model2 = models.simple_model(IMAGE_LEN,IMAGE_WIDTH,NUM_RAGAS, LAMBDA)

    loss = tf.keras.losses.CategoricalCrossentropy()

    opt = Adam(learning_rate = 1e-4)
    '''
    #initialize checkpoint reader object
    chkpt_load = tf.train.load_checkpoint('/fast1/joya.debi/DL/Raag_Classification/tmp/training_checkpts')
    model.load_weights(tf.train.latest_checkpoint('/fast1/joya.debi/DL/Raag_Classification/tmp/training_checkpts'))
    '''

     # Initialize Checkpoint dir and manager
    checkpt_dr = './10raagsckpt'
    checkpt = tf.train.Checkpoint(optimizer = opt, model = model2)
    status = checkpt.restore(tf.train.latest_checkpoint(checkpt_dr))
    status.expect_partial()

    for i, batch in enumerate(iter(list_test)):
        batch = batch.reset_index(drop=True)
        print(batch)
        for index, row in batch.iterrows():
            new_row = batch.iloc[[index]]
            #new_row = pd.DataFrame(row.to_numpy(), columns = ['File path', 'Raga', 'Raga One-Hot'])
            # print(new_row, type(new_row))
            data_x, data_y = preprocessing.create_batch_2(new_row)

            slice_pred = []                                                                                                 
            slice_loss = []        
            
            for j in range(0, len(data_x)):
                # print('Hello')
                prediction = model2(np.expand_dims(np.expand_dims(data_x[j], axis = 2), axis=0))
                loss_value = loss(np.expand_dims(data_y[j], axis =1).T, prediction)
                slice_pred.append(prediction.numpy()[0])
                slice_loss.append(loss_value)


            # print(slice_pred)
            print("="*50)
            testing = np.mean(np.array(slice_pred), axis = 0)
            print(new_row['Raga'], end = ' ')
            print(output_2_rankings(testing, enc, num_display = 5))


            # print(output_2_rankings(prediction.numpy()[0], enc, num_display = 0), loss_value)
            # print(loss_value, type(loss_value))

if __name__ == "__main__":
   main()
