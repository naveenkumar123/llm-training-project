"""
@author: Naveen N G
@date: 28-09-2025
@description: Mobile shop application using RAG technique and Chroma vector database.
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
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
    documents = []
    data_folder_path = os.path.join(current_dir, "data")
    folders = glob.glob(f"{data_folder_path}/*")
    text_encoding = {'encoding': 'utf-8'}
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*", loader_cls=TextLoader, loader_kwargs=text_encoding)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(len(chunks))
    # print(chunks[2])
    return chunks

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

    collection = vector_databse._collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    return result


def visulaize_vectore_store(vector_db_data, visualise_type : str, ):
    vectors = np.array(vector_db_data['embeddings'])
    documents = vector_db_data['documents']
    doc_types = [metadata['doc_type'] for metadata in vector_db_data['metadatas']]
    colors = [['blue', 'green'][['company', 'mobile-price'].index(t)] for t in doc_types]

    dimension =  2 if visualise_type == '2D' else 3
    tsne = TSNE(n_components=dimension, random_state=42, perplexity=len(chunks)-5)
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

chunks = longchain_document_load()
vector_store_data = create_vector_database(chunks)
visulaize_vectore_store(vector_store_data, '3D')


# Use this command to run : python src/rag/mobile-shop/long_chain.py 