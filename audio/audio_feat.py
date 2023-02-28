import os
import librosa
import numpy as np
from panns_inference import AudioTagging, SoundEventDetection, labels
from tqdm import tqdm
import torch

def audio_feature(audio_paths, vid_paths, feature_folder, holder_dir):
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    audio_feature_folder = f"{feature_folder}/audio_feats"
    
    if not os.path.exists(audio_feature_folder):
        os.makedirs(audio_feature_folder)
        
    audio_tagger = AudioTagging(checkpoint_path=None, device=device)
    counter = 0
    for audio_path in tqdm(audio_paths):

        
        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)

        (clipwise_output, embedding) = audio_tagger.inference(audio)

        if counter == 0:
            holder1 = embedding

        else:
            holder1 = np.add(holder1,embedding)
            
        
        name = audio_path.split('.')[0].split('/')[-1]

        audio_feature_path = f"{audio_feature_folder}/{name}_feat.npy"
        
        np.save(audio_feature_path, embedding)

        counter += 1

    holder1 = holder1 / counter
    
    audio_dir = f"{holder_dir}/audio_wavs"
    
    for vid in tqdm(vid_paths):
        video_name = vid.split('/')[-1].split('.')[0]
        holder_path = f"{audio_dir}/{video_name}_audio.wav"
        if not os.path.exists(holder_path):
            audio_feature_path = f"{audio_feature_folder}/{video_name}_audio_feat.npy"
            np.save(audio_feature_path, holder1)
