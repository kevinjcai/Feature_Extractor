import whisper
import torchaudio
import torch



path = "/data/msr_vtt_test/msr_vtt_holder/train_val_videos_audio/video1_audio.wav"
audio_tensor =torch.reshape(torchaudio.load(path)[0],(80,-1))
print(audio_tensor)
print(torch.Tensor.size(audio_tensor))
model = whisper.load_model("base")
# result = model.detect_language(audio_tensor)
# print(result["text"])
a = tras.videomae.feature_extraction_videomae
