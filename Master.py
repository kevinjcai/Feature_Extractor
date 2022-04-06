import sys
import os

def generateVideo_txt(path):
    path = path
    dir_list = os.listdir(path)
    name = path.split('/')[-2]
    folder = path.split('/')[-1]
    for i in range(len(dir_list)):
        dir_list[i] = path + dir_list[i]
    txtPath = '/home/kevincai/Feature_Extractor/' + name + '_holder'
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
            feature_folder = name + '_holder/' + folder + '_features'
            command = "python feat_extract.py --data-list {0} --model \
                i3d_resnet50_v1_kinetics400 --save-dir {1}".format(txtName, feature_folder)
            os.system(command)
    else:
        print(sys.argv)
        raise Exception("Need exactly 1 argument")
