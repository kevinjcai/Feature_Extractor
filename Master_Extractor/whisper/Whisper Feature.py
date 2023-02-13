import whisper
import os
import tqdm
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import numpy
from transformers import BertTokenizer, BertModel
import numpy as np
import tqdm

def whisp_feat():
    model = whisper.load_model("large")
    for i in tqdm.tqdm(range(10000)):
        path = f"/data/msr_vtt_test/msr_vtt_holder/train_val_videos_audio/video{i}_audio.wav"
        vid  = f"video{i}"
        if os.path.exists(path) and not os.path.exists(f'/data/msr_vtt_test/msr_vtt_holder/transcription/{vid}_transcription.txt'):
            # print(path)
            result = model.transcribe(path)
            # print( result["text"],result["language"])
            if result != "":
                with open(f'/data/msr_vtt_test/msr_vtt_holder/transcription/{vid}_transcription.txt', 'w') as f:
                    f.write(result["text"])



# Define the input string
def bert():
    holder1 = np.zeros((5,768),dtype=np.float64)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    for  i in tqdm.tqdm(range(10000)):
        
        path = f"/data/msr_vtt_test/msr_vtt_holder/transcription/video{i}_transcription.txt"
        
        if not os.path.exists(path):
            path = f"/data/msr_vtt_test/msr_vtt_holder/whisper_embeddings/whisper{i}_feature.npy"
            np.save(path, holder1)
            continue
        if os.path.exists(f"/data/msr_vtt_test/msr_vtt_holder/whisper_embeddings/whisper{i}_feature.npy"):
            continue
        with open(path, 'r') as f:
            input_string = ' '.join(f.readlines())


        tokens = tokenizer.tokenize(input_string)

        if tokens == []:
            path = f"/data/msr_vtt_test/msr_vtt_holder/whisper_embeddings/whisper{i}_feature.npy"
            np.save(path, holder1)
            continue
        # Encode the tokens into BERT's input format
        encoded_input = tokenizer.encode(tokens, add_special_tokens=True)

        outputs = model(torch.tensor([encoded_input]))
        sentence_embedding = outputs[0][0]

        
        path = f"/data/msr_vtt_test/msr_vtt_holder/whisper_embeddings/whisper{i}_feature.npy"
        np.save(path, sentence_embedding.detach().numpy().astype(np.float64))

def convert():
    for i in tqdm.tqdm(range(10000)):
        path = f"/data/msr_vtt_test/msr_vtt_holder/whisper_embeddings/whisper{i}_feature.npy"
        arr = np.load(path)
        np.save(path, arr.astype(np.float32))
        
whisp_feat()
bert()   
convert()
