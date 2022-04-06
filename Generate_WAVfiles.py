# import subprocess

# def Audio_from_Video(videoPath, audioPath):
#     command = "ffmpeg -i {0} -vn -acodec copy {1}".format(videoPath, audioPath)
    

#     subprocess.call(command, shell=True)
from moviepy.editor import *
import moviepy.editor as mp
import subprocess
import os
import sys

def Audio_from_Video(video_file, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)
    print(filename, output_ext)
    command = "ffmpeg -i /data/MSVD/YouTubeClips/uqVCk2oDpSE_194_200.avi -ab 160k -ac 2 -ar 44100 -vn audio.wav"
    subprocess.call(command, shell=True)
    # subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
    #                 stdout=subprocess.DEVNULL,
    #                 stderr=subprocess.STDOUT)
# def Audio_from_Video(videoPath, audioPath):
#     audioclip = AudioFileClip(r'%s' % videoPath)
#     print(audioclip)
#     audioclip.write_audiofile(r"my_result.wav")
#     # video.audio.write_audiofile(r"my_result.wav")

video_path = "/data/MSVD/YouTubeClips/uqVCk2oDpSE_194_200.avi"
audio_path = "audio.wav"
Audio_from_Video(video_path)