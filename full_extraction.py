import os
import os.path
import argparse
from utils import create_videotxt, generate_WAV, video_scene_extractor
from video import vid_feat
from audio import audio_feat
from mae import mae_feat
from whispers import whisper_feat
from CLIP import CLIP_feature

def file_paths(path):
    dir_list = os.listdir(path)
    holder = []
    for i in range(len(dir_list)):
        p = path + '/' + dir_list[i]
        if os.path.isdir(p):
            holder = holder + file_paths(p)
        else:
            holder.append(p)
    return holder   

def dir_path(path):
    dir_list = os.listdir(path)
    holder = []
    for i in range(len(dir_list)):
        p = path + '/' + dir_list[i]
        holder.append(p)
    return holder 


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
    
    asset_path = f"{feature_path}/assets"
    if not os.path.exists(asset_path):
        os.makedirs(asset_path)
    
    vid_paths = file_paths(video_path)
    print("-----Wav Extraction Start-----")
    audio_dir = generate_WAV.audio_from_video(vid_paths, asset_path)
    audio_paths = file_paths(audio_dir)
    
    print("-----Frame Extraction Start-----")
    # frame_paths = video_scene_extractor.video_to_images(vid_paths, asset_path)
    frame_paths = "/data/msr_vtt_test/feat_extraction/assets/scene_frames"
    
    frame_dirs = dir_path(frame_paths)
    for i, frames in enumerate(frame_dirs):
        frame_dirs[i] = sorted(file_paths(frames), key=lambda i: int(os.path.splitext(os.path.basename(i))[0][6:]))
        
    
    
    
    txt_path = create_videotxt.generate_txt(vid_paths, asset_path)

    
    all_feats = f"{feature_path}/feats"
    if not os.path.exists(all_feats):
        os.makedirs(all_feats)
    
    # print("-----Video Feat Extraction Start-----")
    # vid_feat.video_feat(txt_path, all_feats)
    # print("-----Audio Feat Extraction Start-----")
    # audio_feat.audio_feature(audio_paths, vid_paths, all_feats, asset_path)
    # print("-----Mae Feat Extraction Start-----")
    # mae_feat.mae_extractor(vid_paths, all_feats)
    print("-----Whisper Feat Extraction Start-----")
    whisper_feat.whisp_feat(audio_paths, vid_paths, all_feats, asset_path)
    # print("-----Clip Feat Extraction Start-----")
    # CLIP_feature.clip_feat(frame_dirs, all_feats)
    
    
# vid_path
# /data/msr_vtt_test/holder_vids
# feat path
# /data/msr_vtt_test/exp_feats
# Python cli
# python full_extraction.py --videos=/data/msr_vtt/train_val_videos --output=/data/msr_vtt_test/feat_ext
# python full_extraction.py --videos=/data/msr_vtt_test/holder_vids --output=/data/msr_vtt_test/_feat
