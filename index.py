"""
Index module for BM25 PDF search system.

This module handles:
1. PDF text extraction
2. Text preprocessing (tokenization, lowercasing, etc.)
3. BM25 index creation
4. Saving and loading index using pickle
"""

import os
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# PDF text extraction libraries
import fitz  # PyMuPDF
import nltk
from rank_bm25 import BM25Okapi

# Download necessary NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class PDFIndexer:
    """Class to handle PDF document indexing with BM25."""
    
    def __init__(self, documents_dir: str, index_dir: str):
        """
        Initialize the PDFIndexer.
        
        Args:
            documents_dir: Directory containing PDF documents
            index_dir: Directory to store index files
        """
        self.documents_dir = Path(documents_dir)
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "bm25_index.pickle"
        self.document_info_path = self.index_dir / "document_info.pickle"
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Store document information and content
        self.document_info = {}  # Maps doc_id to metadata (filename, etc.)
        self.document_contents = {}  # Maps doc_id to full text
        self.document_paragraphs = {}  # Maps doc_id to list of paragraphs
        
        # BM25 index and corpus
        self.bm25_index = None
        self.corpus = []
        self.doc_ids = []
        
        # Indonesian stopwords
        self.stopwords = set(nltk.corpus.stopwords.words('indonesian'))
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, List[str]]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing:
                - Full text of the PDF
                - List of paragraphs in the PDF
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            paragraphs = []
            
            for page in doc:
                text = page.get_text()
                full_text += text
                
                # Split text into paragraphs (non-empty lines)
                page_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                paragraphs.extend(page_paragraphs)
            
            return full_text, paragraphs
        
        except Exception as e:
            print(f"Error extracting text from {pdf_path.name}: {e}")
            return "", []
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing.
        
        Args:
            text: Text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        # Lowercase the text
        text = text.lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [token for token in tokens 
                 if token.isalnum() and token not in self.stopwords]
        
        return tokens
    
    def build_index(self) -> None:
        """
        Build BM25 index from PDF documents in the documents directory.
        """
        start_time = time.time()
        print("Building index...")
        
        # Check if documents directory exists
        if not self.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")
        
        # Get list of PDF files
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {self.documents_dir}")
            return
        
        # Reset data structures
        self.document_info = {}
        self.document_contents = {}
        self.document_paragraphs = {}
        self.corpus = []
        self.doc_ids = []
        
        # Process each PDF file
        for doc_id, pdf_path in enumerate(pdf_files):
            print(f"Processing {pdf_path.name}...")
            
            # Extract text from PDF
            full_text, paragraphs = self.extract_text_from_pdf(pdf_path)
            
            if not full_text:
                print(f"Skipping {pdf_path.name}: No text extracted")
                continue
            
            # Store document information
            self.document_info[doc_id] = {
                'filename': pdf_path.name,
                'path': str(pdf_path),
                'size': os.path.getsize(pdf_path)
            }
            
            # Store document contents
            self.document_contents[doc_id] = full_text
            self.document_paragraphs[doc_id] = paragraphs
            
            # Preprocess text for BM25
            tokens = self.preprocess_text(full_text)
            
            # Add to corpus
            self.corpus.append(tokens)
            self.doc_ids.append(doc_id)
        
        # Create BM25 index
        if self.corpus:
            self.bm25_index = BM25Okapi(self.corpus)
            
            # Save index
            self.save_index()
            
            print(f"Index built successfully in {time.time() - start_time:.2f} seconds.")
            print(f"Indexed {len(self.corpus)} documents.")
        else:
            print("No documents were indexed.")
    
    def save_index(self) -> None:
        """
        Save the BM25 index and document information using pickle.
        """
        # Create a dictionary with all necessary data
        index_data = {
            'bm25_index': self.bm25_index,
            'corpus': self.corpus,
            'doc_ids': self.doc_ids
        }
        
        # Save index data
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Save document information
        doc_data = {
            'document_info': self.document_info,
            'document_contents': self.document_contents,
            'document_paragraphs': self.document_paragraphs
        }
        
        with open(self.document_info_path, 'wb') as f:
            pickle.dump(doc_data, f)
        
        print(f"Index saved to {self.index_path}")
        print(f"Document information saved to {self.document_info_path}")
    
    def load_index(self) -> bool:
        """
        Load the BM25 index and document information from pickle files.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        try:
            # Check if index files exist
            if not self.index_path.exists() or not self.document_info_path.exists():
                print("Index files not found. Run indexing first.")
                return False
            
            # Load index data
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25_index = index_data['bm25_index']
            self.corpus = index_data['corpus']
            self.doc_ids = index_data['doc_ids']
            
            # Load document information
            with open(self.document_info_path, 'rb') as f:
                doc_data = pickle.load(f)
            
            self.document_info = doc_data['document_info']
            self.document_contents = doc_data['document_contents']
            self.document_paragraphs = doc_data['document_paragraphs']
            
            print(f"Index loaded successfully with {len(self.corpus)} documents.")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False


# Command-line interface if run as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index PDF documents for BM25 search")
    parser.add_argument("--documents", default="./documents", help="Path to documents directory")
    parser.add_argument("--index", default="./index", help="Path to index directory")
    
    args = parser.parse_args()
    
    # Create indexer and build index
    indexer = PDFIndexer(documents_dir=args.documents, index_dir=args.index)
    indexer.build_index()
