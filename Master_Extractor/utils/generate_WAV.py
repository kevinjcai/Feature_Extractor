import os
from tqdm import tqdm


def audio_from_video(video_dir, output_path, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    
    audio_folder = f"{output_path}/audio_wavs"
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
        print("The new directory is created!")
    
    for vid in tqdm(video_dir):
        video_name = vid.split('/')[-1].split('.')[0]
        audio_path = f"{audio_folder}/{video_name}_audio.wav"
        if not os.path.exists(audio_path):
            command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -loglevel quiet -vn {1}".format(vid, audio_path)
            os.system(command)
            
    return audio_folder