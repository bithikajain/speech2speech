#
# Copyright (C) 2020 Bithika Jain
#
import os
import math
import numpy as np
import librosa
import librosa.display

# make spectogram array from trimmmed audio files
def create_spectrogram_array():
    for audio_file in file_list:
        samples, sample_rate = librosa.load(audio_file, sr = 16000)
        audio_name = audio_file.split('trim_silence_30db')[1].split('/')[-1].split('.')[0]
        filename  = os.path.join(spectogram_array_path_trim_30db , 'trim_spec'+ '_'+audio_name)
        X = np.abs(librosa.stft(samples))
        np.save(filename, X)


# make spectogram array from trimmmed audio files
def create_spectrogram_array_ntft():
    for audio_file in file_list:
        samples, sample_rate = librosa.load(audio_file, sr = 16384)
        audio_name = audio_file.split('trim_silence_30db')[1].split('/')[-1].split('.')[0]
        filename  = os.path.join(spectogram_array_path_trim_30db_ntft512 , 'trim_spec'+ '_'+audio_name)
        X = np.abs(librosa.stft(samples, ntft = 512))
        np.save(filename, X)
