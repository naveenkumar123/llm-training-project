

"""
@author: Naveen N G
@date: 15-09-2025
@description: This module provides functionality to Image to Digital text using Ollama llama3.2-vision model.. Multi model processing.
Usage:
    Run this file directly to launch the Gradio UI for brochure generation.
"""


# ollama pull llama3.2-vision
import os
import io
import base64
from PIL import Image
import gradio as gr
from IPython.display import Markdown, display, update_display
import ollama

def get_stream_from_image(file_path):
    try:
        # Image to sream base64 encoded string..
        img = Image.open(file_path)
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()
        base64_encoded_string = base64.b64encode(image_bytes).decode('utf-8')
        return base64_encoded_string
    except FileNotFoundError:
        print("Error: Image file not found.")
        exit()


def image_to_text(message, history):
    user_text = message["text"]
    uploaded_files = message["files"]
    imageBase64Encoding = get_stream_from_image(uploaded_files[0])

    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': user_text,
            'images': [imageBase64Encoding]
        }],
        stream =True
    )

    result = ''
    for chunk in response:
        result += chunk.message.content or ""
        yield result
        

    # print(response) 


def render_ui():
    with gr.Blocks() as demo:
        gr.ChatInterface(
            fn=image_to_text,
            multimodal=True,  # Enables file uploads
            title='Welcome to Image processing - Image to Digital text. Multi model processing..',flagging_mode="never"
        )
    demo.launch(share=True)


render_ui()

# To run this file, uncomment the below line and execute the script
# python src/image_to_text/DigitalImage.py
