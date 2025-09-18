"""
@author: Naveen N G
@date: 18-09-2025
@description: Application to Convert aduio into summarized content.
"""



# install torch torchvision torchaudio
# pip install  requests bitsandbytes transformers==4.48.3 accelerate==1.3.0
# brew install ffmpeg


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf
from IPython.display import Markdown, display
from transformers import pipeline, AutoTokenizer, AutoProcessor, TextStreamer, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM
import torch


login_hf()

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

audio_model= 'openai/whisper-medium'
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(audio_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
speech_model.to(device)
processor = AutoProcessor.from_pretrained(audio_model)

def getPipline():
    task_pipline = pipeline(
                    "automatic-speech-recognition",
                    model=speech_model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch.float16,
                    device=device,
                    return_timestamps=True,
                )
    return task_pipline 

def audioToText():
    task_pipline = getPipline()
    current_directory = os.getcwd()
    if 'audio_summary' in current_directory:
        audio_file_path  = os.path.join(current_directory, 'audio-en.mp3')
    else:
        audio_file_path  = os.path.join(current_directory + '/src/hugging_face_models/audio_summary', 'audio-en.mp3')
    result = task_pipline(audio_file_path)
    text = result["text"]
    return text


model_name= 'meta-llama/Meta-Llama-3.1-8B-Instruct'
def summriiseText(transcripts):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    user_prompt = f"""
                Below is an extract transcript of a personal meeting.
                Please write minutes in markdown, including a summary, location and date; discussion points; takeaways; and action items with owners.\n {transcripts}"
                """
    
    messages = [
        {"role": "system", "content": "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."},
        {"role": "user", "content": {user_prompt}}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer)      
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
    result = tokenizer.decode(outputs[0])
    print(result)
    display(Markdown(result))

transcripts = audioToText() # Adutio to text
summriiseText(transcripts)

# Use this command to run : python src/hugging_face_models/audio_summary/AudioSummary.py 