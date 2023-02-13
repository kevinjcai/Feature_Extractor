import cv2
import os

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def video_to_image(path: str):
# Read the video from specified path

    try:
        
        # creating a folder named data
        if not os.path.exists('/data/msr_vtt_test/images_first_frame'):
            os.makedirs('/data/msr_vtt_test/images_first_frame')
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    dir_list = os.listdir(path)
    dir_list.sort(key=natural_keys)
    for i in range(len(dir_list)):
        dir_list[i] = path + '/' + dir_list[i]
    # frame
    for video in dir_list:    
        cam = cv2.VideoCapture(video)
        currentframe = 0
        name = video.split('/')[-1]
        while(True):
            
            # reading from frame
            ret,frame = cam.read()
        
            if ret:
                # if video is still left continue creating images
                name = '/data/msr_vtt_test/images_first_frame/' + name + '.jpg'
                print ('Creating...' + name)
        
                # writing the extracted images
                cv2.imwrite(name, frame)
        
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
                break
            else:
                break
        
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_to_image("/data/msr_vtt/train_val_videos")