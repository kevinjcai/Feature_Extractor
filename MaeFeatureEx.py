from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
import numpy as np
import torch
import cv2
import os
import skvideo.io  

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
feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

for i in range(5875,10000):
    print(f"video_{i} Start")
    videodata = skvideo.io.vread(f'/data/msr_vtt/train_val_videos/video{i}.mp4')  

    vShape = videodata.shape
    width = vShape[2]
    height = vShape[1]

    if width > height:
        videodata = np.pad(videodata, [(0,0),(0,width-height),(0,0),(0,0)], mode="constant")
    elif height > width:
        videodata = np.pad(videodata, [(0,0),(0,0),(0,height-width),(0,0)], mode="constant")

    vShape = videodata.shape

    videodata = list(videodata.reshape((vShape[0],vShape[3],vShape[1],vShape[2])))
    rand = list(range(len(videodata)))
    np.random.shuffle(rand)
    currFrames = rand[:16]
    holder = []
    for j in currFrames:
        holder.append(videodata[j])
    
    
    # videodata = np.random.choice(list(videodata.reshape((vShape[0],vShape[3],vShape[1],vShape[2]))), 16)
    
    video = holder
    num_frames = 16
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
    print(f"video_{i} Done")
    np.save(f"/data/msr_vtt_test/MAE_Features/mae_feature_video{i}.npy", embedding)