# pip install diffusers transformers accelerate bitsandbytes datasets==3.6.0 fsspec==2023.9.2
# pip install torchcodec


# pip install transformers==4.48.3 datasets==3.2.0
from huggingface_hub import login
hf_token = ''


def login_hf():
    login(hf_token)