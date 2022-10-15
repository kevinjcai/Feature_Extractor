# import subprocess

# def Audio_from_Video(videoPath, audioPath):
#     command = "ffmpeg -i {0} -vn -acodec copy {1}".format(videoPath, audioPath)
    

#     subprocess.call(command, shell=True)
from genericpath import exists
import subprocess
import os
import sys
import time
import wave
def Audio_from_Video(path, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    name = path.split('/')[-2]
    folder = path.split('/')[-1]
    audio_folder = name + '_holder/' + folder + '_audio'
    dir_list = os.listdir(path)
    name = path.split('/')[-2]
    folder = path.split('/')[-1]
    audio_folder = '/home/kevincai/Feature_Extractor/' + audio_folder
    isExist = os.path.exists(audio_folder)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(audio_folder)
        print("The new directory is created!")
    
    for i in range(len(dir_list)):
        video = dir_list[i].split('.')
        video_name = video[0]
        video_path = path + '/' + dir_list[i]
        audio_path = audio_folder + '/' + video_name + "_audio.wav"
        if not os.path.exists(audio_path):
            command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(video_path, audio_path)
            subprocess.call(command, shell=True)
        # time.sleep(1)
    # subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
    #                 stdout=subprocess.DEVNULL,
    #                 stderr=subprocess.STDOUT)
# def Audio_from_Video(videoPath, audioPath):
#     audioclip = AudioFileClip(r'%s' % videoPath)
#     print(audioclip)
#     audioclip.write_audiofile(r"my_result.wav")
#     # video.audio.write_audiofile(r"my_result.wav")

video_path = "/data/msr_vtt/msr_vtt/train_val_videos"
Audio_from_Video(video_path)
# /data/msr_vtt/msr_vtt/train_val_videos