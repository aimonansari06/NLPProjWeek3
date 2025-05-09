# retriever.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import os
import PyPDF2  # For PDF support
from typing import List, Dict, Union
import pickle

class Retriever:
    """
    A document retriever that uses sentence embeddings and FAISS for efficient similarity search.
    
    Features:
    - Handles text, markdown (.md), and PDF (.pdf) files
    - Automatic chunking of documents
    - FAISS-based indexing for fast retrieval
    - Persistent storage/loading of indexes
    
    Usage:
    >>> retriever = Retriever()
    >>> retriever.add_documents(["doc1.txt", "doc2.pdf"])
    >>> results = retriever.query("search query", k=3)
    >>> retriever.save("index_folder")
    >>> loaded_retriever = Retriever.load("index_folder")
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the Retriever.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            chunk_size: Number of characters per document chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.metadata = []
        self.index = None
        self.embeddings = None
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - self.chunk_overlap
        return chunks
    
    def _read_file(self, file_path: str) -> str:
        """
        Read content from a file (supports .txt, .md, .pdf).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() for page in reader.pages])
            return text
        else:  # txt or md
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def add_documents(self, documents: Union[List[str], str], metadata: Union[List[Dict], Dict, None] = None):
        """
        Add documents to the retriever. Can be file paths or raw text.
        
        Args:
            documents: List of file paths or text strings
            metadata: Optional metadata for each document
        """
        if isinstance(documents, str):
            documents = [documents]
            
        if metadata is None:
            metadata = [{}] * len(documents)
        elif isinstance(metadata, dict):
            metadata = [metadata] * len(documents)
            
        for doc, meta in zip(documents, metadata):
            # If it's a file path that exists
            if isinstance(doc, str) and os.path.exists(doc):
                text = self._read_file(doc)
                meta['source'] = doc
            else:
                text = doc
                
            chunks = self._chunk_text(text)
            self.documents.extend(chunks)
            self.metadata.extend([meta.copy() for _ in chunks])
            
        # If we have documents but no index, create one
        if self.documents and self.index is None:
            self._create_index()
        # If index exists, add to it
        elif self.documents and self.index is not None:
            self._update_index()
    
    def _create_index(self):
        """Create FAISS index from current documents."""
        self.embeddings = self.model.encode(self.documents)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def _update_index(self):
        """Update existing FAISS index with new documents."""
        new_embeddings = self.model.encode(self.documents[len(self.embeddings):])
        faiss.normalize_L2(new_embeddings)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.index.add(new_embeddings)
    
    def query(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing:
            - 'text': document chunk
            - 'score': similarity score (1 is most similar)
            - 'metadata': associated metadata
        """
        if not self.documents:
            raise ValueError("No documents have been added to the retriever")
            
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # FAISS returns L2 distance - we convert to cosine similarity
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            results.append({
                'text': self.documents[idx],
                'score': 1 - dist,  # Convert L2 distance to cosine similarity
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def save(self, path: str):
        """
        Save the retriever state to disk.
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save other data
        with open(os.path.join(path, "data.pkl"), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embeddings': self.embeddings,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """
        Load a retriever from disk.
        
        Args:
            path: Directory path containing saved retriever
            
        Returns:
            Loaded Retriever instance
        """
        # Initialize with dummy parameters (will be overwritten)
        retriever = cls(chunk_size=512, chunk_overlap=50)
        
        # Load FAISS index
        retriever.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load other data
        with open(os.path.join(path, "data.pkl"), 'rb') as f:
            data = pickle.load(f)
            retriever.documents = data['documents']
            retriever.metadata = data['metadata']
            retriever.embeddings = data['embeddings']
            retriever.chunk_size = data['chunk_size']
            retriever.chunk_overlap = data['chunk_overlap']
        
        return retriever