"""
title: Llama Index Pipeline
author: Maxim Tigulev
date: 22.01.2025
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from PDF-documents using the Llama Index library.
requirements: llama-index==0.12.1, sentence-transformers==3.3.1, mistralai==1.1.0
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index import Document
from llama_index import pdf_loader

# Global debug mode variable
debug_mode = True

# Get API-KEY from OS Variable
api_key = os.getenv("MISTRAL_API_KEY")
if debug_mode:
    print(f"API Key: {api_key}")
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Main Functions
def read_pdf(file_path: str):
    doc = pdf_loader(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ===== Function to call LLM  =====
def call_llm(query, rag_text):
    system_prompt = f"You are professional analyst in bank card busines and respond to user's questions about documents. The most relevant parts of the document provided here: {rag_text}"

    if debug_mode:
        print("### Call LLM ")
        print(f"User query: {query}")

    full_prompt = f"{system_prompt}\n{query}"

    try:
        response = client.chat.complete(
        model=model,
            messages = [
                {
                    "role": "user",
                    "content":  full_prompt
                },
            ]
            )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(f"Request failed: {e}. Please check your request.")
        return None

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        # Specify the file path and name
        file_path = "./data/vau_users_guide.pdf"
        file_name = "vau_users_guide.pdf"

        if debug_mode:
            print(f"Reading PDF from: {file_path}")

        # Read the PDF file
        text = read_pdf(file_path)

        # Create a Document object
        document = Document(text=text, extra_info={"file_name": file_name})

        # Create a list of documents
        self.documents = [document]

        # Create the index
        self.index = VectorStoreIndex.from_documents(self.documents)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        if debug_mode:
            print(messages)
            print(user_message)

        # Create the query engine
        query_engine = self.index.as_query_engine(streaming=True)

        # Find relevant pages for user_message
        rag_text = query_engine.query(user_message)

        # Call the LLM with user_query and rag_text
        response = call_llm(user_message, rag_text)

        return response
