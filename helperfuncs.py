import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    raags= ['recordings/Abhogee', 'recordings/Bhageshri', 'recordings/Bhoop/', 'recordings/Bhairav/']



if __name__ == "__main__":
    main()
