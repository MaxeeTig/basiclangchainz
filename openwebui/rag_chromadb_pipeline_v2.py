"""
title: Chromadb RAG Pipeline
author: Maxim Tigulev
date: 23.01.2024
version: 1.0
license: MIT
requirements: PyPDF2, sentence-transformers, chromadb, langchain_mistralai, langchain_core
description: A pipeline for retrieving relevant information from a knowledge base using chromadb.
"""
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Union, Generator, Iterator
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

# Global debug mode variable
debug_mode = True

class Pipeline:
    def __init__(self):
        self.name = "Document RAG Search V2"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatMistralAI(model="mistral-large-latest")
        self.dim = 384
        self.client = chromadb.Client(Settings(chroma_server_host= "chromadb-engine",
                                chroma_server_http_port="8000"
                                ))
        self.collection = None
        self.text_chunks = {}

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
        self.store_embeddings(embeddings)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

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

    def store_embeddings(self, embeddings):
        # Create a ChromaDB collection
        self.collection = self.client.create_collection(name="pdf_embeddings")

        # Generate unique IDs for each embedding
        ids = [str(i) for i in range(len(embeddings))]

        # Insert embeddings into the collection
        self.collection.add(embeddings=embeddings.tolist(), ids=ids)

    def get_text_by_ids(self, ids):
        return [self.text_chunks[id] for id in ids[0]]

    def generate_response(self, user_message, similar_texts):
        system_prompt = f'''
        As helpful assistant on the first line of customer support you answer the questions of users on documents related to
        bank card operations and payments.
        Relevant information provided here {similar_texts}, user query provided here {user_message}

        #Actions#
        1. Carefully review user message.
        2. Study relevant information.
        3. Prepare extended answer to user on the basis of qu.

        #Output:#
        Response to user query
        '''
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_message),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if self.collection is None:
            self.on_startup()
        query_embedding = self.model.encode([user_message])
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=5)
        similar_texts = self.get_text_by_ids(results['ids'])
        response = self.generate_response(user_message, similar_texts)
        return response
