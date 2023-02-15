import os
import os.path
import argparse
from utils import create_videotxt, generate_WAV, video_scene_extractor

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

def video_feat(txt_name, feature_path, gpu = -1):
    vid_feat_path = f"{feature_path}/vid_feats"
    extracter_path = "/home/kevincai/Feature_Extractor/Master_Extractor/video/feat_extract.py"
    command = "python {0} --data-list {1} --model i3d_resnet50_v1_kinetics400 --save-dir {2} --gpu-id={3}".format(extracter_path, txt_name, vid_feat_path, gpu)
    os.system(command)


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
    
    audio_dir = generate_WAV.audio_from_video(vid_paths, feature_path)
    audio_paths = dir_paths(audio_dir)
    
    frame_paths = video_scene_extractor.video_to_images(vid_paths, feature_path)
    
    txt_path = create_videotxt.generate_txt(vid_paths, feature_path)

    
    all_feats = f"{feature_path}/feats"
    if not os.path.exists(all_feats):
        os.makedirs(all_feats)
    
    video_feat(txt_path, all_feats)
     