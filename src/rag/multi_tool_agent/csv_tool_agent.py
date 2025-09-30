

"""
@author: Naveen N G
@date: 29-09-2025
@description: CSV tool agent.
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
import glob
from typing import List
import pandas as pd
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents import Tool, AgentType, AgentExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
load_dotenv()


def csv_tool_agent():
    data_folder_path = os.path.join(parent_dir, "mobile_shop", "data", "mobile-price")
    folders = glob.glob(f"{data_folder_path}/*")
    all_df = []
    for csv_file in folders:
        df = pd.read_csv(csv_file)
        all_df.append(df)

    # Create the Multi-DataFrame Agent 
    ollama_chat = ChatOllama(model="llama3.2", temperature=0, num_gpu=8)
   
    # Create a specialized agent for the DataFrame, using the Ollama chat model
    csv_base_agent = create_pandas_dataframe_agent(
        ollama_chat,
        df,
        verbose=False,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True
    )

    csv_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=csv_base_agent.agent,
        tools=csv_base_agent.tools,
        verbose=False,
        max_iterations=100,
    )

    # Wrap the CSV agent as a single tool
    csv_tool = Tool(
        name="csv_data_tool",
        func=csv_agent_executor.run,
        description="""
            This tool is useful for questions about Model price details which should include: Model name, Color, Memory, Storage, Rating, Selling Price, Original Price.
            Use ONLY this tool for any question involving NUMERICAL calculation, comparison, or data analysis 
            on the Selling Price and Original Price.
            Do not make anything up if you haven't been provided with relevant context
            """,
    )
    print('CSV Agent is ready..')
    return csv_tool

# csv_tool_agent()

