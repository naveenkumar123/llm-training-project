"""
@author: Naveen N G
@date: 20-09-2025
@description: This application converts MySql queries and schema to PostgreSQL.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hugging_face_models.Login import login_hf
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from IPython.display import Markdown, display
import gradio as gr

login_hf()

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

model_name='mistralai/Mixtral-8x7B-Instruct-v0.1'

def query_convertion(query):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

    system_prompt = """
                        You are an assistant and expert in RDBMS Database. You have to rewrite the MySQL queries and schema to highly optimised PostreSQL queries and schema.
                        Do not provide any explanation and comments, consider the data types.
                """
    user_prompt = f"""
                 Rewrite the MySQL queries, schema to PostgreSQL queries and schema, provide highly optimised queries. Respond with queries and schema, 
                do not provide any explanation.\n\n
                {query}
                """
    messages = [
       { "role" : "system", "content": system_prompt },
       { "role" : "user", "content": user_prompt }
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    result = model.generate(inputs, max_new_tokens=500)
    output = tokenizer.decode(result[0])
    print(output)
    return output

def render_ui():
    with gr.Blocks() as ui:
        gr.Markdown("Convert Convertion ToolL: Python to Javascript")
        with gr.Row():
            mySql = gr.Textbox(label="MySql Queries/Schema:", lines=10)
            postgreSql = gr.Textbox(label="PostgreSql Queries/Schema:", lines=10)
        with gr.Row():
            convert = gr.Button("Convert code")   
        convert.click(query_convertion, inputs=[mySql], outputs=[postgreSql])     
    ui.launch(inbrowser=True)

render_ui()