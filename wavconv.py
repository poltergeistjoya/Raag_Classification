import os

from os import path
from pydub import AudioSegment

raags = ['Bhoopali/', 'Bhairav/']
for i in raags:
    mp3files = os.scandir(i)
    for mp3 in mp3files:
        title = mp3.name.split(".")
        src = mp3.name
        dst = i+ title[0]+'.wav'

        #convert wav to mp3
        sound = AudioSegment.from_mp3(i+src)
        sound.export(dst, format = "wav")
        #print(dst)
