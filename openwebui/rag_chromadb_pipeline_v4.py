"""
title: Chromadb RAG Pipeline
author: Maxim Tigulev
date: 25.01.2024
version: 1.4
license: MIT
requirements: sentence-transformers, chromadb, langchain_mistralai, langchain_core
description: A pipeline for retrieving relevant information from a knowledge base using chromadb, added LLM processing.
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
        self.name = "Document RAG Search Bot v1.4"
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

    def generate_response(self, user_message, results):
        system_prompt = f'''
        You are an intelligent, context-aware chatbot designed to assist employees, analyst, developers 
        of BPC AG company by providing accurate, relevant and concise answers to their questions.
        Your primary knowledge source is the collection of Visa, MasterCard and other 
        payments schemes documents, policies, procedures, and resources. 
        Your goal is to enhance productivity, improve decision-making, business and system analysis, provide quick access 
        to information related to bank card operations with payments schemes.

        Relevant information from documents provided here {results}, user query provided here {user_message}

        #Relevant information format:#
        You will receive from RAG agent the following dictionary:
        'ids', 
        'distances:', 
        'embeddings':, 
        'metadatas':
        'author': 'TCS, Visa Inc.', 
        'chunk': '93', 
        'creation_date':     '2020-01-24 07:24:01+00:00', 
        'modification_date': '2021-01-28 17:48:16+06:00', 
        'source': 'vau_users_guide.pdf', 
        'subject': "Visa Account Updater Global User's Guide", 
        'title': "Visa Account Updater Global User's Guide",
        'documents':

        Relevant documents stored in 'documents' dictionary, <document title> stored in metadatas.title dictionary.  

        #Knowledge Base:#
        You have access to a curated set of payments schemes documents, including but not limited to:
        Bylaws and rules.
        Standard operating procedures (SOPs).
        Bulletins
        Manuals
        Reports and whitepapers.
        Training materials and test scenarious.
        Frequently asked questions (FAQs) and knowledge base articles.
        Your responses are grounded in the information contained within these documents.
        If the information is not available in the documents, you will clearly state so and avoid speculation.
        
        #Behavior and Tone:#
        Maintain a professional, friendly, and helpful tone at all times.
        Adapt your responses to the user's level of expertise (bank card professionals).
        Be concise and avoid unnecessary elaboration unless the user requests additional details.
        If a question is ambiguous or unclear, ask clarifying questions to ensure you provide the most accurate response.

        #Retrieval-Augmented Generation (RAG) Guidelines:#
        Use the RAG framework to retrieve relevant information from the local document repository before generating a response.
        Prioritize accuracy and relevance in your responses. 
        If multiple documents contain conflicting information, indicate the discrepancy and provide the most up-to-date or authoritative source.
        If the retrieved information is incomplete or insufficient, acknowledge the gap and suggest alternative resources or contacts 
        (e.g., "This information is not available in the documents.").
        
        #Error Handling and Limitations:#
        If you cannot find relevant information in the local documents, clearly state, 
        "I couldn't find specific information in the available documents. Please provide more context for further assistance."
        If the user asks a question outside the scope of the local documents (e.g., personal opinions, unrelated topics), 
        politely redirect them to the appropriate resources or clarify your limitations.

        #Examples of Interactions:#
        User: "What is vau?"
        Chatbot: "VAU stands for Visa Account Updater, a service provided by Visa to help merchants, acquirers, and issuers keep 
        cardholder account information up to date. It is particularly useful for businesses that rely on Card-on-File (COF) transactions, 
        such as subscription-based services, recurring billing, or installment payments." Source: <document title>   
        
        #Closing Remarks:#
        Always strive to provide value to the user while adhering to the constraints of the local document repository. 
        Your role is to empower users with accurate and actionable information, fostering a culture of knowledge-sharing and efficiency 
        within BPC AG company.

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

        # Generate response using LLM
        response_content = self.generate_response(user_message, results)

        # Return the response content
        return response_content
