"""
@author: Naveen N G
@date: 20-09-2025
@description: This application converts Python code into Javascript.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from IPython.display import Markdown, display
import gradio as gr

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

model_name='mistralai/Mixtral-8x7B-Instruct-v0.1'

def code_convertion(pythonCode):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    system_prompt = """
                        You are an assistant and expert in Python and Javascript. You have to rewrite the python code in high performing javascript code, it sould be compitable with NodeJS.
                        Do not provide any explanation and comments.
                """


    user_prompt = f"""
                Rewrite the python code to javascript code, provide highly optimised code. Respond with only the javascript code, 
                do not provide any explanation of the code.\n\n
                {pythonCode}
                """

    messages = [
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": user_prompt }
    ]

    
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    # streamer = TextStreamer(tokenizer)      
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    result = model.generate(inputs, max_new_tokens=400)
    result = tokenizer.decode(result[0])
    print(result)
    display(Markdown(result))


    
def renderUi():
    with gr.Blocks() as ui:
        gr.Markdown("Convert Convertion ToolL: Python to Javascript")
        with gr.Row():
            python = gr.Textbox(label="Python code:", lines=10)
            javascript = gr.Textbox(label="Javascript code:", lines=10)
        with gr.Row():
            convert = gr.Button("Convert code")   
        convert.click(code_convertion, inputs=[python], outputs=[javascript])     
    ui.launch(inbrowser=True)


renderUi()

# Use this command to run : python src/hugging_face_models/code_convertion/PythonToJavascript.py 