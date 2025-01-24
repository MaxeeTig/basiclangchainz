"""
title: Chromadb RAG Pipeline
author: Maxim Tigulev
date: 23.01.2024
version: 1.0
license: MIT
requirements: sentence-transformers, chromadb, langchain_mistralai, langchain_core
description: A pipeline for retrieving relevant information from a knowledge base using chromadb.
"""
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Union, Generator, Iterator
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Global debug mode variable
debug_mode = True

class Pipeline:
    def __init__(self):
        self.name = "Document RAG Search V3"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatMistralAI(model="mistral-large-latest")
        self.client = chromadb.HttpClient(host="chromadb-engine", port="8000")
        self.collection = self.client.get_collection("documents_embeddings")

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def get_text_by_ids(self, chunks_file_path, ids):
        with open(chunks_file_path, 'r') as f:
            chunks = json.load(f)
        return [chunks[int(id)] for id in ids]

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
            return "Error: Collection is not set."
        if debug_mode:
            print(f"Collection name: {self.collection.name}")

        query_embedding = self.model.encode([user_message])
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=5)

        if debug_mode:
            print(f"Query results: {results}")

        relevant_ids = results['ids'][0]
        relevant_texts = self.get_text_by_ids('./data/vau_users_guide_chunks.json', relevant_ids)
        joined_text = ' '.join(relevant_texts)
        return joined_text
