import clip
import torch

from torchvision import transforms
import os
from PIL import Image
import numpy
from tqdm import tqdm
# print(clip.available_models())
def create_clip_features(path: str):
    dir_list = os.listdir(path) 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('ViT-L/14@336px', device = device)
    basedir = os.path.dirname(path)
    if not os.path.exists("/data/msr_vtt_test/CLIP_Features/"):
        os.makedirs("/data/msr_vtt_test/CLIP_Features/")
    count =0 
    for image_path in tqdm(dir_list):
        img = Image.open(f"{path}/{image_path}")
        if count == 0:
            print(f"/data/msr_vtt_test/CLIP_Features/clip_feature_{image_path}.npy")

        img_tensor = preprocess(img).unsqueeze(0).to(device)
        clip_feature = model.encode_image(img_tensor).cpu().detach().numpy()
        clip_feature = clip_feature.astype(numpy.single)

        count += 1
        file_path = f"/data/msr_vtt_test/CLIP_Features/clip_feature_{image_path}.npy"
        try:
            a = open(file_path, "x")    
        except:
            pass
        numpy.save(f"/data/msr_vtt_test/CLIP_Features/clip_feature_{image_path}.npy", clip_feature)


if __name__ == "__main__":
    create_clip_features("/data/msr_vtt_test/images_first_frame")