"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
requirements: fitz, frontend, tools, annoy
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
"""
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from typing import List, Dict, Union, Generator, Iterator
import numpy as np
#from mistralai import Mistral
import os

# Global debug mode variable
debug_mode = True

class Pipeline:
    def __init__(self):
        self.name = "Document RAG Search"
        self.text_chunker = lambda text, chunk_size: [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.embedding_generator = SentenceTransformer('all-MiniLM-L6-v2')
        dimension = 384  # Dimension of the embeddings generated by SentenceTransformer
        self.annoy_index = AnnoyIndex(dimension, 'angular')
        self.metadata_dict = {}
        self.chunks = []

    async def on_startup(self):
        # Get API-KEY from OS Variable
        #api_key = os.getenv("MISTRAL_API_KEY")
        #model = "mistral-large-latest"
        #client = Mistral(api_key=api_key)

        # Initialize the necessary models and components

        # === Initialize Annoy index and metadata dictionary ===

        file_path = "./data/vau_users_guide.pdf"

        if debug_mode:
            print(f"Debug: Reading doc: {file_path}")

        # read PDF file
        doc_text = self.read_pdf(file_path)

        if debug_mode:
            print("Debug: Creating chunks...")

        # split to chunks by 1000 symbols
        self.chunks = self.chunk_text(doc_text, 1000)

        # for each chunk
        chunk_id = 0
        for chunk in self.chunks:
            # create embedding
            embeddings = self.generate_embeddings(chunk)
            # put index in dictionary
            metadata = {"source": file_path, "chunk": f'{chunk_id}'}
            self.metadata_dict[chunk_id] = metadata
            # add embedding to Annoy index
            self.store_embeddings(chunk_id, embeddings)
            chunk_id += 1

        self.annoy_index.build(10)  # 10 trees

        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def read_pdf(self, file_path: str):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()

        return text

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        return self.text_chunker(text, chunk_size)

    def generate_embeddings(self, text: str) -> List[float]:
        return self.embedding_generator.encode(text).tolist()

    def store_embeddings(self, chunk_id: int, embeddings: List[float]) -> None:
        self.annoy_index.add_item(chunk_id, embeddings)

    def embed_query(self, query: str) -> List[float]:
        return self.generate_embeddings(query)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(f"Debug: {messages}")
        print(f"Debug: {user_message}")

        # convert user query to vector
        query_embedding = self.embed_query(user_message)

        # Search in the vector DB for the nearest neighbors
        k = 5  # Number of nearest neighbors to retrieve
        indices = self.annoy_index.get_nns_by_vector(query_embedding, k)

        print(f"Indices: {indices}")

        # join chunks with defined indexes into rag_text
        rag_text = ""
        for index in indices:
            rag_text += self.chunks[index] + "\n"

        return rag_text
