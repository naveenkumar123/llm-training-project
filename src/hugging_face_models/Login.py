# pip install diffusers transformers accelerate bitsandbytes datasets==3.6.0 fsspec==2023.9.2
# pip install torchcodec


# pip install transformers==4.48.3 datasets==3.2.0
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from huggingface_hub import login


load_dotenv()


def login_hf():
    HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
    login(HUGGING_FACE_TOKEN)