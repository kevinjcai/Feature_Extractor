import whisper
import os
import tqdm
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

def dir_paths(path):
    dir_list = os.listdir(path)
    holder = []
    for i in range(len(dir_list)):
        p = path + '/' + dir_list[i]
        if os.path.isdir(p):
            holder = holder + dir_paths(p)
        else:
            holder.append(p)
    return holder

def whisp_feat(audio_paths, vid_paths, feature_folder, holder_dir):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = whisper.load_model(name="large",device=device)
    trans_folder = f"{holder_dir}/transcriptions"
    
    if not os.path.exists(trans_folder):
        os.makedirs(trans_folder)
        
    for audio in tqdm(audio_paths):
        
        
            # print(path)
        result = model.transcribe(audio)
        # print( result["text"],result["language"])
        if result != "":
            audio_name = audio.split('/')[-1].split('.')[0]

            transciption_path = f"{trans_folder}/{audio_name}_transcription.txt"
            with open(transciption_path, 'w') as f:
                f.write(result["text"])
    
    
    holder1 = np.zeros((5,768),dtype=np.float64)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    
    whisp_folder = f"{feature_folder}/whisper_feat"
    
    if not os.path.exists(whisp_folder):
        os.makedirs(whisp_folder)
        
    for vid in tqdm(vid_paths):
        video_name = vid.split('/')[-1].split('.')[0]
        wisp_path = f"{whisp_folder}/{video_name}_whisper_feat_.npy"
        
        path = f"{trans_folder}/{video_name}_audio_transcription.txt"
        if not os.path.exists(path):
            
            np.save(wisp_path, holder1)
            continue
        
            
        with open(path, 'r') as f:
            input_string = ' '.join(f.readlines())


        tokens = tokenizer.tokenize(input_string)

        if tokens == []:
            np.save(wisp_path, holder1)
            continue
        # Encode the tokens into BERT's input format
        encoded_input = tokenizer.encode(tokens, add_special_tokens=True)

        outputs = model(torch.tensor([encoded_input]))
        sentence_embedding = outputs[0][0]

        
        np.save(wisp_path, sentence_embedding.detach().numpy().astype(np.float64))


    # for i in tqdm.tqdm(range(10000)):
    #     path = f"/data/msr_vtt_test/msr_vtt_holder/whisper_embeddings/whisper{i}_feature.npy"
    #     arr = np.load(path)
    #     np.save(path, arr.astype(np.float32))
        

