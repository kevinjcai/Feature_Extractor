from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining, VideoMAEFeatureExtractor
import numpy as np
import torch
import cv2
import os
import skvideo.io 
from tqdm import tqdm 

# num_frames = 16
# video = list(np.random.randn(16, 3, 224, 224))

# feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
# model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

# pixel_values = feature_extractor(video, return_tensors="pt").pixel_values
# print(pixel_values.shape)
# num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
# seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
# bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

# outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
# loss = outputs.loss


# num_frames = 16
# video = list(np.random.randn(16, 3, 224, 224))

# feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
# model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

# pixel_values = feature_extractor(video, return_tensors="pt").pixel_values
# print(pixel_values.shape)
# num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
# seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
# bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

# outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
# loss = outputs.loss


def mae_extractor(video_paths, feature_folder):
    feature_extractor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
    
    mae_feature_folder = f"{feature_folder}/mae_feats"
    
    if not os.path.exists(mae_feature_folder):
        os.makedirs(mae_feature_folder)
        
    for vid in tqdm(video_paths):
        videodata = skvideo.io.vread(vid)  

        vShape = videodata.shape
        width = vShape[2]
        height = vShape[1]

        if width > height:
            videodata = np.pad(videodata, [(0,0),(0,width-height),(0,0),(0,0)], mode="constant")
        elif height > width:
            videodata = np.pad(videodata, [(0,0),(0,0),(0,height-width),(0,0)], mode="constant")

        vShape = videodata.shape

        num_frames = 16

        
        videodata = list(videodata.reshape((vShape[0],vShape[3],vShape[1],vShape[2])))
        rand = list(range(len(videodata)))
        np.random.shuffle(rand)
        currFrames = rand[:num_frames]
        holder = []
        for j in currFrames:
            holder.append(videodata[j])
        
        
        # videodata = np.random.choice(list(videodata.reshape((vShape[0],vShape[3],vShape[1],vShape[2]))), 16)
        
        video = holder
        # model.config.num_frames = num_frames
        # model.config.image_size = vShape[2]
        # print(model.config)
        pixel_values = feature_extractor(video, return_tensors="pt").pixel_values
        num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
        model.config.output_hidden_states = True
        seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
        bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()
        outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

        embedding = outputs.hidden_states[0][0].detach().numpy()
        
        video_name = vid.split('/')[-1].split('.')[0]

        np.save(f"{mae_feature_folder}/{video_name}_mae_feat.npy", embedding)