import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import OneHotEncoder
import time

from os import path
from pydub import AudioSegment
from joblib import Memory

memory = Memory(".cache")
'''
Given a list of directories with  .mp3 files, all will  be converted to wav
*** needs some extra functionality to delete the .mp3 files after***
'''
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

# Function to plot the converted audio signal
def plot_spec(D, sr, raag = "raag"):
    fig, ax = plt.subplots(figsize = (30,10))
    spec = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title = 'Spectrogram of ' + raag)
    fig.colorbar(spec)
    plt.show()


def generate_dataset(dataset_path = "./raga-data"):
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
    raga_dict = dict()
    raga_directories = next(os.walk(dataset_path))[1]

    for raga_directory in raga_directories:
        recordings_path = os.path.join(dataset_path, raga_directory)
        filenames = [os.path.join(recordings_path, x).replace("\\","/") for x in next(os.walk(recordings_path), (None, None, []))[2]]
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


def load_wav_16k_mono(filename, sampling_rate = 16000):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    # https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels = 1 # skip second channel
    )
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out = sampling_rate)
    return wav

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

    #plot_spec(batch_x[100], target_sr)

    return batch_x, batch_y

@memory.cache()
def create_batch_2(dataset, use_chroma = False, n_fft = 2048):
    '''
    To conserve memory, the dataset will only consist of a list of filenames and the raga they correspond too.
    At runtime, this function will be called periodically to retrieve the next batch of audio data.

    Parameters:
        - n_fft = FFT window size # https://github.com/librosa/librosa/issues/1194
    '''
    batch_x_mels = []
    batch_x_chromas = []
    batch_y = []

    dataset = dataset.reset_index()
    duration = 30.0
    target_sr = 8000
    target_size = duration * target_sr

    for index, audiofile in dataset.iterrows():
        t = time.time()
        filename = audiofile['File path']
        file_length = round(librosa.get_duration(filename = audiofile['File path']), 0)

        sr = librosa.get_samplerate(filename)
        stream = librosa.stream(filename,
                            block_length=256,
                            frame_length=2048,
                            hop_length=2048)
        # print(stream)
        mels = []
        chromas = []
        blocks = []

        for i, y_block in enumerate(stream):
<<<<<<< HEAD
            print(i, end = ' ')

=======
            # print(i, end = ' ')
>>>>>>> 739ec47055ac83677a78e177d7a9e9e9851366f2
            m_block = librosa.feature.melspectrogram(y=y_block, sr=sr,
                                                    n_fft=n_fft,
                                                    hop_length=2048,
                                                    center=False)
            mels.append(m_block)

            # Forget all this, load in the entire file, and then split it up and take chromas from there!
            if use_chroma:
<<<<<<< HEAD
                if (i % 4 == 0) and (i != 0):
                    input = np.concatenate( blocks, axis=0 )    # https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
                    chroma = librosa.feature.chroma_cqt(y=input, sr=sr)
                    #S = np.abs(librosa.stft(y_block, n_fft=n_fft))**2
                    #chroma = librosa.feature.chroma_stft(y=S, sr=sr)
                    chromas.append(chroma)
                    blocks = y_block
                    print("Here!")
                    blocks = []
                    blocks.append(y_block)
                else:
                    blocks.append(y_block)


=======
                blocks.append(y_block)

        if use_chroma:
            input = np.concatenate(blocks, axis=0 )    # https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
            S = np.abs(librosa.stft(input, n_fft=4096))**2
            chroma = librosa.feature.chroma_stft(y=S, sr=sr)
            # chroma = librosa.feature.chroma_cqt(y=input, sr=sr)
            chromas.append(chroma)
   
>>>>>>> 739ec47055ac83677a78e177d7a9e9e9851366f2
        # Add data to batch
        batch_x_mels.extend(mels)
        batch_x_chromas.extend(chromas)
        batch_y.extend([audiofile['Raga One-Hot'] for i in range(len(mels))])

        print(f'Elapsed: {round(time.time() - t, 2)}')

    if use_chroma:
<<<<<<< HEAD
        return {"mels": batch_x_mels, "chromas": batc_x_chromas}, batch_y
=======
        return {"mels": batch_x_mels, "chromas": batch_x_chromas}, batch_y 
>>>>>>> 739ec47055ac83677a78e177d7a9e9e9851366f2
    else:
        return batch_x_mels, batch_y

def plot_chroma(signal, sampling_rate):
    S = np.abs(librosa.stft(y, n_fft=4096))**2

    chroma = librosa.feature.chroma_stft(y=S, sr=sampling_rate)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].label_outer()
    #img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
    #fig.colorbar(img, ax=[ax[1]])
    plt.show()


def main():
    testing_stream = False


    data, encoder = generate_dataset()
    print(data)
    print()
    x, y = create_batch_2(data.iloc[0:10], use_chroma = True)
    print(x['mels'][0])
    print(x['chromas'][0])
    print(y[0])
    print(len(x['mels']), len(x['chromas']), len(y))
    print(x['chromas'][0].shape, x['chromas'][78].shape)

    if testing_stream:
        filename1 = 'raga-data\Bageshree\C R Vyas - Raga Bageshri & Chandrakauns.mp3'
        filename2 = 'raga-data\Bageshree\Jagdish Prasad - Raga Bhageshri.mp3'
        filename3 = 'raga-data\Bageshree\Rashid Khan - Raga Bageshri.mp3'
        for filename in [filename1, filename2, filename3]:
            sr = librosa.get_samplerate(filename)
            stream = librosa.stream(filename,
                                block_length=256,
                                frame_length=2048,
                                hop_length=2048)

            mels = []
            for y_block in stream:
                m_block = librosa.feature.melspectrogram(y=y_block, sr=sr,
                                                        n_fft=2048,
                                                        hop_length=2048,
                                                        center=False)
                mels.append(m_block)

            print(len(mels), mels[0])

        print(type(mels))
        mel_sgram = librosa.amplitude_to_db(mels[22], ref=np.min)
        librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    # plot_spec(mel_sgram, sr)
    #y, sr = librosa.load(testing_wav, sr=None)
    #plot_chroma(y, sr)
    '''
    print(y.shape, sr)
    D = to_decibles(y)
    print(D)

    plot_spec(D, sr, raag = "raag")
    '''

if __name__ == "__main__":
    main()
