import os
import os.path


def generate_txt(video_dirs, feature_path):
    txtName = f"{feature_path}/videos.txt"
    with open(txtName, 'w') as filehandle:
        for listitem in video_dirs:
            filehandle.write('%s\n' % listitem)
    return txtName


