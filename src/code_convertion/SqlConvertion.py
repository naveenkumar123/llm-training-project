"""
@author: Naveen N G
@date: 20-09-2025
@description: This application converts MySql queries and schema to PostgreSQL.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from llmchat.GptChat import GptChat
from IPython.display import Markdown, display
import gradio as gr

load_dotenv()



def query_convertion(query):

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

    response = GptChat().chat(messages)
    return response

def render_ui():
    with gr.Blocks() as ui:
        gr.Markdown("DB Convertion: MySql to PostgreSql")
        with gr.Row():
            mySql = gr.Textbox(label="MySql Queries/Schema:", lines=10)
            postgreSql = gr.Textbox(label="PostgreSql Queries/Schema:", lines=10)
        with gr.Row():
            convert = gr.Button("Convert code")   
        convert.click(query_convertion, inputs=[mySql], outputs=[postgreSql])     
    ui.launch(inbrowser=True)

render_ui()


# Use this command to run : python src/code_convertion/SqlConvertion.py 