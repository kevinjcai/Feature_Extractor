import sys
import os
import os.path
import subprocess
import argparse
from video import create_videotxt

def dir_paths(path):
    dir_list = os.listdir(path)
    holder = []
    for i in range(len(dir_list)):
        p = path + '/' + dir_list[i]
        if os.path.isdir(p):
            holder = holder + dir_paths(p)
        else:
            holder.append(p)
    return holder   

def video_feat(vid_dir, feature_path):
    pass
    # folder = path.split('/')[-1]
    # feature_folder = folder_path + name + '_holder/' + folder + '_features' #change to determine where video features are stored
    # command = "python feat_extract.py --data-list {0} --model \
    #     i3d_resnet50_v1_kinetics400 --save-dir {1}".format(txtName, feature_folder)
    # os.system(command)
    # audio_path = Audio_from_Video(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained model")
    parser.add_argument("--videos", type=str, required=True, help="Path to the video directory")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

   
    video_path = args.videos
    feature_path = args.output
    if not os.path.exists(video_path):
        raise Exception("Video Path does not exist")
    elif not os.path.isdir(video_path):
        raise Exception("Video Path is not a directory")
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    if not os.path.isdir(feature_path):
        raise Exception("Feature path is not a directory")
    
    vid_paths = dir_paths(video_path)
    create_videotxt.generate_txt(vid_paths, feature_path)
    
    
     

