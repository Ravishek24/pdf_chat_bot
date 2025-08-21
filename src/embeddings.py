from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
from langchain.docstore.document import Document
import pickle

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
    
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Create embeddings for document chunks"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def build_vector_store(self, documents: List[Document]):
        """Build FAISS vector store from documents"""
        self.documents = documents
        embeddings = self.create_embeddings(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Find most similar documents to query"""
        if self.index is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant documents
        relevant_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                relevant_docs.append(self.documents[idx])
        
        return relevant_docs
    
    def save_vector_store(self, filepath: str):
        """Save vector store to disk"""
        if self.index is None:
            raise ValueError("No vector store to save")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load_vector_store(self, filepath: str):
        """Load vector store from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load documents
            with open(f"{filepath}.pkl", 'rb') as f:
                self.documents = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vector store files not found at {filepath}")