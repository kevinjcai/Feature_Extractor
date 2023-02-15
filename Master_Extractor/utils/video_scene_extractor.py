import cv2
import os
from tqdm import tqdm

from scenedetect import detect, ContentDetector


def video_to_images(video_paths, feature_path):
    all_frame_path = f"{feature_path}/scene_frames"
    if not os.path.exists(all_frame_path):
        os.makedirs(all_frame_path)
        
    for vid in tqdm(video_paths):
        
        scene_list = detect(vid, ContentDetector())
        frames = {0 : 1}
        
        
        for scene in scene_list:
            s0, s1 = scene[0].get_frames(), scene[1].get_frames()
            frames[s0], frames[s1] = 1, 1

        capture = cv2.VideoCapture(vid)
        name = vid.split('.')[0].split('/')[-1]
        frameNr = 0
        scene = 0
        frame_path = f"{all_frame_path}/{name}_frames"
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        while (True):
            success, frame = capture.read()
        
            if success:
                if frameNr in frames:
                    cv2.imwrite(f"{frame_path}/scene_{scene}.jpg", frame)
                    scene += 1
            else:
                break

            frameNr = frameNr+1
    
        capture.release()
        cv2.destroyAllWindows()
    
    return all_frame_path
    
