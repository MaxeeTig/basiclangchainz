"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
"""
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
#from transformers import pipeline
from typing import List, Dict, Union, Generator, Iterator
import numpy as np
from mistralai import Mistral
import os
#from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Global debug mode variable
debug_mode = True

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None
        self.name = "Document RAG Search"

    async def on_startup(self):
        import os

        # Get API-KEY from OS Variable
        api_key = os.getenv("MISTRAL_API_KEY")
        model = "mistral-large-latest"
        client = Mistral(api_key=api_key)

        # Initialize the necessary models and components
        self.pdf_loader = fitz.open
        #text_extractor = fitz.Document
        self.text_chunker = lambda text, chunk_size: [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.embedding_generator = SentenceTransformer('all-MiniLM-L6-v2')
        #vector_store = chromadb.Client()
        #llm_integrator = pipeline('text2text-generation', model='google/flan-t5-base')

        # === Initialize FAISS index and metadata dictionary ===
        dimension = 384  # Dimension of the embeddings generated by SentenceTransformer
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.metadata_dict = {}

        #from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        if debug_mode:
            print("Debug: Reading docs...")

        #self.documents = SimpleDirectoryReader("./data").load_data()

        if debug_mode:
            print(f"Documents loaded: {len(self.documents)}")

        if debug_mode:
            print("Debug: Creating index...")

        #self.index = VectorStoreIndex.from_documents(self.documents)

        if debug_mode:
            print(f"Index created: {self.index is not None}")

        # Print the size of the read documents
        total_size = sum(len(doc.text) for doc in self.documents)
        print(f"Total size of read documents: {total_size} characters")

        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def read_pdf(self, file_path: str):
        doc = self.pdf_loader(file_path)
        text = ""
        for page in doc:
            text += page.get_text()

        return text

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        return self.text_chunker(text, chunk_size)

    def generate_embeddings(self, text: str) -> List[float]:
        return self.embedding_generator.encode(text).tolist()

    def store_embeddings(self, embeddings: List[float], metadata: Dict) -> None:
        self.faiss_index.add(np.array([embeddings]).astype('float32'))

    def embed_query(self, query: str) -> List[float]:
        return self.generate_embeddings(query)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(f"Debug: {messages}")
        print(f"Debug: {user_message}")

        user_input = "Name main steps of Merchant Onboarding process to mPay?"

        file_path = "mobile-push.pdf"

        # read PDF file
        doc_text = self.read_pdf(file_path)

        # split to chunks by 1000 symbols
        chunks = self.chunk_text(doc_text, 1000)

        # for each chunk
        chunk_id = 0
        for chunk in chunks:
            # create embedding
            embeddings = self.generate_embeddings(chunk)
            # put index in dictionary
            metadata = {"source": file_path, "chunk": f'{chunk_id}'}
            self.metadata_dict[chunk_id] = metadata
            # add embeddin to FAISS index
            self.store_embeddings(embeddings, metadata)
            chunk_id += 1

        # convert user query to vector
        query_embedding = self.embed_query(user_input)

        # Search in the vector DB for the nearest neighbors
        k = 5  # Number of nearest neighbors to retrieve
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)

        print(f"Distances: {distances}")
        print(f"Indices: {indices}")

        # join chunks with defined indexes into rag_text
        rag_text = ""
        for index in indices[0]:
            rag_text += chunks[index] + "\n"

        #query_engine = self.index.as_query_engine(streaming=True)
        #response = query_engine.query(user_message)

        return rag_text
