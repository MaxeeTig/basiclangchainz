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
        self.name = "Document RAG Search V3"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatMistralAI(model="mistral-large-latest")
        self.dim = 384
        self.client = chromadb.HttpClient(host="chromadb-engine", port="8000")
        self.collection = "documents_embeddings"
        self.text_chunks = {}

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

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
