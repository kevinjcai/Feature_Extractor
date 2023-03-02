import clip
import torch

from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def clip_feat(frame_paths, feature_folder):
     

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('ViT-L/14@336px', device = device)
    
    clip_feature_folder = f"{feature_folder}/clip_feats"
    
    if not os.path.exists(clip_feature_folder):
        os.makedirs(clip_feature_folder)
        

    
    for image_path in tqdm(frame_paths):
        name = image_path[0].split('/')[-2]
        clip_features = []
        for path in image_path:
            img = Image.open(path)

            img_tensor = preprocess(img).unsqueeze(0).to(device)
            clip_feature = model.encode_image(img_tensor).cpu().detach().numpy()
            clip_features.apppend(clip_feature.astype(np.single))
            
        np.save(f"{clip_feature_folder}/{name}_clip.npy", np.stack(clip_features, axis = 0))


