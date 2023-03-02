import os
import csv
from time import sleep
dataset_name = "MSR-VTT_"
area = "train_val_videos_"


 
# open .tsv file
def path_to_list(path):
    with open(path) as file:
        
        # Passing the TSV file to 
        # reader() function
        # with tab delimiter
        # This function will
        # read data from file
        tsv_file = csv.reader(file, delimiter="\t")
        holder = []
        for i in tsv_file:
            holder.append(i)
            # print(i)
            # sleep(1)
        # printing data line by line

        return holder

if __name__ == "__main__":
    captions = path_to_list("/data/msr_vtt_test/msr_vtt_videos_holder.tsv")
    
    path = "/data/msr_vtt/train_val_videos"
    
    dir_list = os.listdir(path)
    audioFolder = "/data/msr_vtt_test/msr_vtt_holder/train_val_videos_audio_features"
    audio_list = os.listdir(audioFolder)
    # print(audio_list)
    videosFolder = "/data/msr_vtt_test/msr_vtt_holder/train_val_videos_features"
    videos_list = os.listdir(videosFolder)
    for i in range(len(dir_list)):
        dir_list[i] = dir_list[i].split('.')[0]
    valid_vids = [captions[0]]
    counter = 1
    dir_list.sort()
    dir_list = sorted(dir_list, key = lambda x: (len (x), x))
    # print(dir_list)
    print(videosFolder)
    print(audioFolder)
    for vid in dir_list:
        # audioFeat = f"{vid}_audio.wav_features.npy"
        # print(vid)
        # print(counter)
        # print(captions[counter+1])

        valid_vid = []
        valid_vid.append(dataset_name + area + vid)
        valid_vid.append(captions[counter][1])
        num = vid[5:]
        # print(vid,num)
        if int(num) < 6513:
            valid_vid.append(f"msrvtt-train+{vid}_motion_features.npy")
            valid_vid.append(f"msrvtt-train+{vid}_image_features.npy")
        elif int(num) >= 6513 and int(num) < 7010:
            valid_vid.append(f"msrvtt-eval+{vid}_motion_features.npy")
            valid_vid.append(f"msrvtt-eval+{vid}_image_features.npy")
        elif int(num) >= 7010:
            valid_vid.append(f"msrvtt-test+{vid}_motion_features.npy")
            valid_vid.append(f"msrvtt-test+{vid}_image_features.npy")
        valid_vid.append(f"i3d_resnet50_v1_kinetics400_{vid}.mp4_feat.npy")
        valid_vid.append(f"{vid}_frames_clip.npy")
        valid_vid.append(f"mae_feature_{vid}.npy")
        valid_vid.append(f"{vid}_audio.wav_features.npy")
        valid_vid.append(f"whisper{num}_feature.npy")

        valid_vid.append(captions[counter][-1])
        # print(valid_vid)
        valid_vids.append(valid_vid)
        counter += 1
    nameTSV = "New_ClipAudioWhisp_train_test"
    # with open(f'/data/msr_vtt_test/msr_vtt_holder/msr_vtt_train_val_videos_{nameTSV}.tsv', 'wt') as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     for vids in valid_vids:
    #         tsv_writer.writerow(vids)
    os.makedirs(os.path.dirname(f'/data/msr_vtt_test/msr_vtt_holder/{nameTSV}_tsv/'), exist_ok=True)

    with open(f'/data/msr_vtt_test/msr_vtt_holder/{nameTSV}_tsv/{nameTSV}_features_train.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for vids in valid_vids[:6514]:
            tsv_writer.writerow(vids)

    with open(f'/data/msr_vtt_test/msr_vtt_holder/{nameTSV}_tsv/{nameTSV}_features_val.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for vids in [valid_vids[0]] + valid_vids[6514:7011]:
            tsv_writer.writerow(vids)

    with open(f'/data/msr_vtt_test/msr_vtt_holder/{nameTSV}_tsv/{nameTSV}_features_test.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for vids in [valid_vids[0]] + valid_vids[7011:]:
            tsv_writer.writerow(vids)

# video_id:str	category:int	motion_features:npy(prefix="/data/msr_vtt/mysfire/msrvtt/motion_resnext101_kinetics/",pad=True)	image_features:npy(prefix="/data/msr_vtt/mysfire/msrvtt/image_resnet101/",pad=True)	video_features:npy(prefix="/data/msr_vtt_test/msr_vtt_holder/train_val_videos_features/",pad=True)	clip_features:npy(prefix="/data/msr_vtt_test/CLIP_Features/",pad=True)	maev_feature:npy(prefix="/data/msr_vtt_test/MAE_Features/",pad=True)	audio_feature:npy(prefix="/data/msr_vtt_test/msr_vtt_holder/train_val_videos_audio_features/",pad=True)	caption:nlp.tokenizer(tokenizer="/home/kevincai/mirage/mirage-tokenizers/msrvtt-0.0.2.json", delimiter="|", sample_single_string=True)
