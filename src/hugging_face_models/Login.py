# pip install -q diffusers transformers accelerate bitsandbytes datasets==3.6.0 fsspec==2023.9.2
# pip install torchcodec

from huggingface_hub import login
hf_token = ''


def login_hf():
    login(hf_token)