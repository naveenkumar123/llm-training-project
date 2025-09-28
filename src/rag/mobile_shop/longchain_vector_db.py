"""
@author: Naveen N G
@date: 28-09-2025
@description: RAG technique and Chroma vector database for Mobile shop application.
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
import glob
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

database_name = 'database/shop-vector-db'

def longchain_document_load():
    # documents = []
    all_chunks_and_docs: List[Document] = []
    data_folder_path = os.path.join(current_dir, "data")
    folders = glob.glob(f"{data_folder_path}/*")
    text_encoding = {'encoding': 'utf-8'}
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = None
        if doc_type == 'company':
            # Using TextLoader
            loader = DirectoryLoader(folder, glob="**/*", loader_cls=TextLoader, loader_kwargs=text_encoding)
            # text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            # folder_docs = loader.load()
            # processed_docs = text_splitter.split_documents(folder_docs)
        else:
            # Using CSVLoader
            loader = DirectoryLoader(folder, glob="**/*", loader_cls=CSVLoader, loader_kwargs={'csv_args': {'delimiter': ','}})
            # folder_docs = loader.load()
            # processed_docs = folder_docs
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        folder_docs = loader.load()
        processed_docs = text_splitter.split_documents(folder_docs)
        for doc in processed_docs:
            doc.metadata["doc_type"] = doc_type
            all_chunks_and_docs.append(doc)


    print(len(all_chunks_and_docs))
    # print(all_chunks_and_docs[2])
    return all_chunks_and_docs

def create_vector_database(chunks):
    # embeddings = OpenAIEmbeddings()
    # To use the free emdeddings form Huggingface, use the below code.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    database_path = os.path.join(current_dir, database_name)
    if os.path.exists(database_path):
        # If the database already exists, cleaning the database so that it won't add the same thing to the database
        Chroma(persist_directory=database_path, embedding_function=embeddings).delete_collection()
    
    vector_databse = Chroma.from_documents(documents = chunks, embedding=embeddings, persist_directory=database_path)
    print(f"vector database created with {vector_databse._collection.count()} documents")

    return vector_databse




def visulaize_vectore_store(vector_db_data, perplexity, visualise_type : str, ):
    vectors = np.array(vector_db_data['embeddings'])
    documents = vector_db_data['documents']
    doc_types = [metadata['doc_type'] for metadata in vector_db_data['metadatas']]
    colors = [['blue', 'green'][['company', 'mobile-price'].index(t)] for t in doc_types]

    dimension =  2 if visualise_type == '2D' else 3
    tsne = TSNE(n_components=dimension, random_state=42, perplexity=30, method='exact' ) # remove method='exact' incase of large data
    print(len(vectors))
    reduced_vectors = tsne.fit_transform(vectors)
    fig = None
    if visualise_type == '2D':
        # Create the 2D scatter plot
        fig = go.Figure(data=[go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])
        fig.update_layout(
            title='2D Chroma Vector Database Visualization',
            scene=dict(xaxis_title='x',yaxis_title='y'),
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )       
    else:
        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])
        fig.update_layout(
            title='3D Chroma Vector Database Visualization',
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )  
    fig.write_html(f"{current_dir}/database/vector_database_visualise_{visualise_type}.html", auto_open=True)     
    # fig.show()

def get_shop_vector_data_store():
    chunks = longchain_document_load()
    vector_databse = create_vector_database(chunks)
    return vector_databse

def render_db_visualize():
    chunks = longchain_document_load()
    vector_databse = create_vector_database(chunks)
    collection = vector_databse._collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    perplexity = len(chunks)-5
    visulaize_vectore_store(result, perplexity, '3D')

# render_db_visualize()

# Uncomment the above code to run this to visualize the vector database 2D and 3D diagram. Make sure, you comment the above code when you run the mobile_shop_rag_pipeline
# Use this command to run : python src/rag/mobile_shop/longchain_vector_db.py 