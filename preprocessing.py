import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tfio

from os import path
from pydub import AudioSegment

#given a list of directories with  .mp3 files, all will  be converted to wav
def wavconv(directories):
    for i in directories:
        mp3files = os.scandir(i)
        for mp3 in mp3files:
            title = mp3.name.split(".")
            src = mp3.name
            dst = i+ title[0]+'.wav'

            #convert wav to mp3
            sound = AudioSegment.from_mp3(i+src)
            sound.export(dst, format = "wav")
            #print(dst)
    return 0

def to_decibles(signal):
    # Perform short time Fourier Transformation of signal and take absolute value of results
    stft = np.abs(librosa.stft(signal))
    # Convert to dB
    D = librosa.amplitude_to_db(stft, ref = np.max) # Set reference value to the maximum value of stft.
    return D # Return converted audio signal

# Function to plot the converted audio signal
def plot_spec(D, sr, instrument):
    fig, ax = plt.subplots(figsize = (30,10))
    spec = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title = 'Spectrogram of ' + instrument)
    fig.colorbar(spec)

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    # https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels = 1 # skip second channel
    )
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def create_batch(dataset):
    '''
    To conserve memory, the dataset will only consist of a list of filenames and the raga they correspond too.
    At runtime, this function will be called periodically to retrieve the next batch of audio data 
    '''
    batch_x = []
    batch_y = []

    for audiofile in dataset:
        # Create the absolute path to the file given the relative filename

        # Load the audio in to a np array 

        # Resample the audio to a consistent sampling rate, pad/truncate as needed

        # Any sort of data augmentation (optional - I don't think we need though)
        print(audiofile)

def generate_dataset():
    '''Iterate through the '''
    dataset_path = './raga-data'


def main():
    raags= ['recordings/Abhogee', 'recordings/Bhageshri', 'recordings/Bhoop/', 'recordings/Bhairav/']



if __name__ == "__main__":
    main()
