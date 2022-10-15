import sys
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels


folder_path = '/data/msr_vtt_test/'


def generateVideo_txt(path):
    path = path
    dir_list = os.listdir(path)
    name = path.split('/')[-2]
    folder = path.split('/')[-1]
    for i in range(len(dir_list)):
        dir_list[i] = path + '/' + dir_list[i]
    txtPath = folder_path + name + '_holder'
    isExist = os.path.exists(txtPath)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(txtPath)
        print("The new directory is created!")
    txtName = txtPath + '/' + folder + '_videos.txt'
    with open(txtName, 'w') as filehandle:
        for listitem in dir_list:
            filehandle.write('%s\n' % listitem)
    return txtName

def Audio_from_Video(path, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    name = path.split('/')[-2]
    folder = path.split('/')[-1]
    audio_folder = name + '_holder/' + folder + '_audio'
    dir_list = os.listdir(path)
    
    audio_folder = folder_path + audio_folder #change to determine where audio is stored
    print(audio_folder)
    isExist = os.path.exists(audio_folder)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(audio_folder)
        print(audio_folder)
        print("The new directory is created!")
    
    for i in range(len(dir_list)):
        video = dir_list[i].split('.')
        video_name = video[0]
        print(video)
        video_path = path + '/' + dir_list[i]
        audio_path = audio_folder + '/' + video_name + "_audio.wav"
        print(audio_path)
        command = "ffmpeg -i {0} -hide_banner -loglevel error -ab 160k -ac 2 -ar 44100 -vn {1} ".format(video_path, audio_path)
        os.system(command)
    
    return audio_folder
        # time.sleep(1)
    # subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
    #                 stdout=subprocess.DEVNULL,
    #                 stderr=subprocess.STDOUT)
# def Audio_from_Video(videoPath, audioPath):
#     audioclip = AudioFileClip(r'%s' % videoPath)
#     print(audioclip)
#     audioclip.write_audiofile(r"my_result.wav")
#     # video.audio.write_audiofile(r"my_result.wav")

def Audio_Feature(path):
    device = 'cuda' # 'cuda' | 'cpu'
    # path = 'resources/R9_ZSCveAHg_7s.wav'
    # path = '/data/msr_vtt/msr_vtt/train_val_videos_audio'


    # print('------ Audio tagging ------')

    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""
    name = path.split('/')[-2]
    folder = path.split('/')[-1]
    audio_feature_folder = folder_path + name + '/' + folder + '_features/'
    isExist = os.path.exists(audio_feature_folder)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(audio_feature_folder)
        print(audio_feature_folder)
        print("The new directory is created!")
    dir_list = os.listdir(path)
    name_list = []
    for i in range(len(dir_list)):
        name_list.append(dir_list[i])
        dir_list[i] = path + '/' + dir_list[i]

    at = AudioTagging(checkpoint_path=None, device=device)
    for i in range(len(dir_list)):
        holder = os.path.splitext(name_list[i])
        audio_feature_path = audio_feature_folder + name_list[i] + '_features'
        (audio, _) = librosa.core.load(dir_list[i], sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)    
        (clipwise_output, embedding) = at.inference(audio) 
        np.save(audio_feature_path, embedding)
        print(audio_feature_path)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
        if not os.path.exists(os.path.dirname(path)):
            print(path)
            raise Exception("Directory does not exist")
        else:
            txtName = generateVideo_txt(path=path)
            name = path.split('/')[-2]
            folder = path.split('/')[-1]
            feature_folder = folder_path + name + '_holder/' + folder + '_features' #change to determine where video features are stored
            command = "python feat_extract.py --data-list {0} --model \
                i3d_resnet50_v1_kinetics400 --save-dir {1}".format(txtName, feature_folder)
            os.system(command)
            # audio_path = Audio_from_Video(path)
            # Audio_Feature(audio_path)


            
    else:
        print(sys.argv)
        raise Exception("Need exactly 1 argument")
