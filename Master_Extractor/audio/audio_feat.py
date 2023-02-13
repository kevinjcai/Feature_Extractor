from genericpath import exists
import subprocess
import os
import sys
import numpy as np
import time
import wave
from panns_inference import AudioTagging, SoundEventDetection, labels
import panns_inference
from tqdm import tqdm

def Audio_Feature(path):
    device = 'cuda' # 'cuda' | 'cpu'
    # path = 'resources/R9_ZSCveAHg_7s.wav'
    # path = '/data/msr_vtt/msr_vtt/train_val_videos_audio'


    # print('------ Audio tagging ------')

    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""
    name = path.split('/')[-2]
    folder = path.split('/')[-1]
    audio_feature_folder = folder_path + name + '/' + folder + '_features/'
    isExist = os.path.exists(audio_feature_folder)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(audio_feature_folder)
        print(audio_feature_folder)
        print("The new directory is created!")
    dir_list = os.listdir(path)
    name_list = []
    for i in range(len(dir_list)):
        name_list.append(dir_list[i])
        dir_list[i] = path + '/' + dir_list[i]

    at = AudioTagging(checkpoint_path=None, device=device)
    holder1 = 0
    counter = 0
    # for i in tqdm(range(len(dir_list))):
    for i in tqdm(range(len(dir_list))):

        holder = os.path.splitext(name_list[i])
        audio_feature_path = audio_feature_folder + name_list[i] + '_features'
        (audio, _) = librosa.core.load(dir_list[i], sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)    
        clipwise_output, embedding = at.inference(audio) 
        if counter == 0:
            holder1 = embedding
            # print(embedding)
            # exit(1)
        else:
            # print(embedding[0].shape())
            holder1 = np.add(holder1,embedding)
        np.save(audio_feature_path, embedding)
        # print(audio_feature_path)
        # print(holder1)
        counter += 1

    holder1 = holder1 / counter
    for i in tqdm(range(10000)):
        path = f"/data/msr_vtt_test/msr_vtt_holder/train_val_videos_audio_features/video{i}_audio.wav_features.npy"
        if not os.path.exists(path):
            np.save(path, holder1)
