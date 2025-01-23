import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.config import Settings

# Global debug mode variable
debug_mode = True

class RAGChromaDBApplication:
    def __init__(self, pdf_path, model_name='all-MiniLM-L6-v2', dim=384):
        self.pdf_path = pdf_path
        self.model = SentenceTransformer(model_name)
        self.dim = dim
        self.client = chromadb.Client(Settings(chroma_server_host= "localhost",
                                chroma_server_http_port="8000"
                                ))
        self.collection = None
        self.text_chunks = {}

    def initialize_index(self):
        if not os.path.exists(self.pdf_path):
            print(f"Error: File {self.pdf_path} does not exist.")
            return

        # Read and chunk the PDF
        if debug_mode:
            print(f"Debug: reading file: {self.pdf_path}")
        text_chunks = self.read_pdf(self.pdf_path)

        if debug_mode:
            print(f"Debug: creating chunks: {self.pdf_path}")
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

    def search_similar_docs(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
        similar_texts = self.get_text_by_ids(results['ids'])
        print("Similar Texts:")
        for text in similar_texts:
            print(text)
        return results

if __name__ == "__main__":
    pdf_path = './data/vau_users_guide.pdf'
    app = RAGChromaDBApplication(pdf_path)
    app.initialize_index()
    query = "what is vau?"
    results = app.search_similar_docs(query)
    print(results)
