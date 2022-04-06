import os

path = "/home/kevincai/MSVD/YouTubeClips"
dir_list = os.listdir(path)

for i in range(len(dir_list)):
    dir_list[i] = path + dir_list[i]

with open('video.txt', 'w') as filehandle:
    for listitem in dir_list:
        filehandle.write('%s\n' % listitem)