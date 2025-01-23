"""
title: Chromadb RAG Pipeline
author: Maxim Tigulev
date: 23.01.2024
version: 1.0
license: MIT
requirements: PyPDF2, sentence-transformers, chromadb
description: A pipeline for retrieving relevant information from a knowledge base using chromadb.
"""
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Union, Generator, Iterator

# Global debug mode variable
debug_mode = True

class Pipeline:
    def __init__(self):
        self.name = "Document RAG Search"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384
        self.client = chromadb.Client(Settings(chroma_server_host= "chromadb-engine",
                                chroma_server_http_port="8000"
                                ))
        self.collection = None
        self.text_chunks = {}
        self.collection_name = "pdf_embeddings"

    async def on_startup(self):
        pdf_path = './data/vau_users_guide.pdf'
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} does not exist.")
            return

        # Read and chunk the PDF
        if debug_mode:
            print(f"Debug: reading file: {pdf_path}")
        text_chunks = self.read_pdf(pdf_path)

        if debug_mode:
            print(f"Debug: creating chunks: {pdf_path}")
        chunked_texts = self.chunk_text(text_chunks)

        # Store text chunks with IDs
        self.text_chunks = {str(i): chunk for i, chunk in enumerate(chunked_texts)}

        # Generate embeddings
        if debug_mode:
            print("Debug: generate embeddings.")
        embeddings = self.generate_embeddings(chunked_texts)

        if debug_mode:
            print("Debug: storing embeddings.")
        # Store embeddings in ChromaDB
        self.add_embeddings(embeddings)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass


    def get_or_create_collection(self, name):
            try:
                collection = self.client.get_collection(name=name)
                print(f"Collection '{name}' already exists.")
                return collection, True
            except Exception as e:
                print(f"Collection '{name}' does not exist. Creating a new collection.")
                collection = self.client.create_collection(name=name)
                return collection, False
        

    def read_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text

    def chunk_text(self, text, chunk_size=512, overlap=128):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings

    
    def add_embeddings(self, embeddings):
        ids = [str(i) for i in range(len(embeddings))]    
        if not self.collection_exists:
            try:
                self.collection.add(embeddings=embeddings.tolist(), ids=ids)
                print("Embeddings added successfully.")
            except Exception as e:
                print(f"Error adding embeddings: {e}")
        else:
            print("Collection already exists. Embeddings will not be added.")        

    def get_text_by_ids(self, ids):
        return [self.text_chunks[id] for id in ids[0]]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        query_embedding = self.model.encode([user_message])
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=5)
        similar_texts = self.get_text_by_ids(results['ids'])
        return ' '.join(similar_texts)
