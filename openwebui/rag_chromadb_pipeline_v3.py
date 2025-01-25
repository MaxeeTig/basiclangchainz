"""
title: Chromadb RAG Pipeline
author: Maxim Tigulev
date: 25.01.2024
version: 1.3
license: MIT
requirements: sentence-transformers, chromadb, langchain_mistralai, langchain_core
description: A pipeline for retrieving relevant information from a knowledge base using chromadb.
"""
import os
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Union, Generator, Iterator
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

# Global debug mode variable
debug_mode = True

class Pipeline:
    def __init__(self):
        self.name = "Document RAG Search v1.3"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatMistralAI(model="mistral-large-latest")
        self.client = chromadb.HttpClient(host="chromadb-engine", port="8000")
        self.collection_name = "pdf_embeddings"
        self.collection = None

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def get_text_by_ids(self, ids):
        # Flatten the list of lists
        flat_ids = [id for sublist in ids for id in sublist]
        return [self.collection.get(ids=[id])[0] for id in flat_ids]

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
        if debug_mode:
            print(f"Starting pipe function with user_message: {user_message}")

        # Check if ChromaDB client is initiated
        try:
            if self.client.heartbeat():
                print("Chromadb connection established")
            else:
                raise Exception("Chromadb connection failed")
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"

        # Check if collection exists
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Collection {self.collection_name} exists")
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"

        # Generate query embedding
        query_embedding = self.model.encode([user_message])

        # Search user's query in ChromaDB vector database
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=5)

        if debug_mode:
            print(f"Query results: {results}")

        # Process results - get texts of chunks stored in 'documents' and concatenate them to single string
        similar_texts = ' '.join(results['documents'][0])

        if debug_mode:
            print(f"Similar texts: {similar_texts}")

        # Return the concatenated string
        return similar_texts
