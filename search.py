"""
Search module for BM25 PDF search system.

This module handles:
1. Loading the BM25 index using pickle
2. Preprocessing search queries
3. Searching documents using the BM25 algorithm
4. Returning the top 3 most relevant documents with snippets
"""

import os
import time
import re
from typing import List, Dict, Tuple, Any
from pathlib import Path

import nltk
from rank_bm25 import BM25Okapi

# Import our indexer module
from index import PDFIndexer

class PDFSearcher:
    """Class to handle BM25 search of indexed PDF documents."""
    
    def __init__(self, documents_dir: str = "./documents", index_dir: str = "./index"):
        """
        Initialize the PDFSearcher.
        
        Args:
            documents_dir: Directory containing PDF documents
            index_dir: Directory where index files are stored
        """
        self.indexer = PDFIndexer(documents_dir=documents_dir, index_dir=index_dir)
        self.index_loaded = False
    
    def load_index(self) -> bool:
        """
        Load the BM25 index.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        # Load index using the indexer
        self.index_loaded = self.indexer.load_index()
        return self.index_loaded
    
    def build_index_if_needed(self) -> None:
        """Build index if it doesn't exist or can't be loaded."""
        if not self.load_index():
            print("Building index...")
            self.indexer.build_index()
            self.index_loaded = True
    
    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess search query.
        
        Args:
            query: Search query text
        
        Returns:
            List of preprocessed query tokens
        """
        return self.indexer.preprocess_text(query)
    
    def find_best_snippet(self, doc_id: int, query_tokens: List[str]) -> str:
        """
        Find the best matching snippets from a document for a query.
        
        Args:
            doc_id: Document ID
            query_tokens: Preprocessed query tokens
            
        Returns:
            Up to 5 most relevant sentences containing query tokens
        """
        paragraphs = self.indexer.document_paragraphs.get(doc_id, [])
        if not paragraphs:
            return "No snippet available"
        
        # Convert query tokens to a set for faster lookup
        query_token_set = set(query_tokens)
        
        # Split paragraphs into sentences
        all_sentences = []
        import nltk.tokenize
        for paragraph in paragraphs:
            sentences = nltk.tokenize.sent_tokenize(paragraph)
            all_sentences.extend(sentences)
        
        # Score each sentence based on query token matches
        scored_sentences = []
        for sentence in all_sentences:
            # Preprocess sentence
            sentence_tokens = self.indexer.preprocess_text(sentence)
            
            # Count matching tokens
            match_count = sum(1 for token in sentence_tokens if token in query_token_set)
            
            # Store sentences with at least one match
            if match_count > 0:
                scored_sentences.append((sentence, match_count))
        
        # Sort sentences by match count (descending)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Get up to 5 best matching sentences
        top_snippets = scored_sentences[:5]
        
        # If no sentences match, return the first substantial sentence
        if not top_snippets and all_sentences:
            for s in all_sentences:
                if len(s.split()) >= 5:
                    return s
            if all_sentences:
                return all_sentences[0]
            
        # Join the top snippets with separator
        if top_snippets:
            return " [...] ".join([snippet for snippet, _ in top_snippets])
        else:
            return "No relevant snippets found"
    
    def highlight_matches(self, text: str, query_tokens: List[str]) -> str:
        """
        Highlight matching query terms in text.
        
        Args:
            text: Text to highlight (can be multiple snippets separated by [...])
            query_tokens: Query tokens to highlight
            
        Returns:
            Text with matches highlighted (in terminal using ANSI colors)
        """
        # If text contains multiple snippets (separated by [...]), highlight each one separately
        if " [...] " in text:
            snippets = text.split(" [...] ")
            highlighted_snippets = [self._highlight_snippet(snippet, query_tokens) for snippet in snippets]
            return " [...] ".join(highlighted_snippets)
        else:
            return self._highlight_snippet(text, query_tokens)
    
    def _highlight_snippet(self, text: str, query_tokens: List[str]) -> str:
        """
        Highlight matching query terms in a single snippet of text.
        
        Args:
            text: Text snippet to highlight
            query_tokens: Query tokens to highlight
            
        Returns:
            Text with matches highlighted (in terminal using ANSI colors)
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        result = text
        
        # ANSI escape codes for bold text
        bold_start = "\033[1m"
        bold_end = "\033[0m"
        
        # Keep track of adjustments to text positions due to inserted highlighting
        offset = 0
        
        # Sort query tokens by length (longest first) to avoid highlighting partial matches
        for token in sorted(query_tokens, key=len, reverse=True):
            # Escape regex special characters
            pattern = re.escape(token)
            
            # Find all occurrences of the token in text (case-insensitive)
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                
                # Adjust for previous insertions
                adj_start = start + offset
                adj_end = end + offset
                
                # Replace the original text with highlighted version
                original_match = result[adj_start:adj_end]
                highlighted = f"{bold_start}{original_match}{bold_end}"
                
                # Add highlighting markers to the result
                result = result[:adj_start] + highlighted + result[adj_end:]
                
                # Update the offset for subsequent matches
                offset += len(highlighted) - len(original_match)
                
                # Mark this part as processed in text_lower to avoid double-highlighting
                text_lower = text_lower[:start] + "#" * (end - start) + text_lower[end:]
        
        return result
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing:
                - doc_id: Document ID
                - filename: Document filename
                - score: BM25 relevance score
                - snippet: Best matching snippet
        """
        # Ensure index is loaded
        if not self.index_loaded:
            self.build_index_if_needed()
        
        # Preprocess query
        query_tokens = self.preprocess_query(query)
        
        if not query_tokens:
            return []
        
        # Perform BM25 search
        scores = self.indexer.bm25_index.get_scores(query_tokens)
        
        # Get document IDs and scores
        doc_scores = [(self.indexer.doc_ids[i], scores[i]) for i in range(len(scores))]
        
        # Sort by score (descending)
        sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Get top K results
        top_results = []
        for doc_id, score in sorted_results[:top_k]:
            # Skip documents with 0 score
            if score <= 0:
                continue
            
            # Get document information
            doc_info = self.indexer.document_info.get(doc_id, {})
            
            # Find best snippet
            snippet = self.find_best_snippet(doc_id, query_tokens)
            # Highlight query terms in snippet
            highlighted_snippet = self.highlight_matches(snippet, query_tokens)
            
            # Add to results
            top_results.append({
                'doc_id': doc_id,
                'filename': doc_info.get('filename', f"Document {doc_id}"),
                'score': score,
                'snippet': highlighted_snippet
            })
        
        return top_results
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for display.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted string of search results
        """
        if not results:
            return "No relevant documents found."
        
        output = f"Top {len(results)} most relevant documents:\n"
        output += "-" * 80 + "\n"
        
        for i, result in enumerate(results):
            output += f"{i+1}. {result['filename']} (Score: {result['score']:.2f})\n"
            output += f"   Relevant snippets:\n"
            
            # Display snippets with better formatting
            snippets = result['snippet'].split(" [...] ")
            for j, snippet in enumerate(snippets):
                output += f"     • {snippet.strip()}\n"
                
            output += "-" * 80 + "\n"
        
        return output


# Command-line interface if run as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search PDF documents using BM25")
    parser.add_argument("--documents", default="./documents", help="Path to documents directory")
    parser.add_argument("--index", default="./index", help="Path to index directory")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--top", type=int, default=3, help="Number of top results to return")
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = PDFSearcher(documents_dir=args.documents, index_dir=args.index)
    
    # Build index if needed
    searcher.build_index_if_needed()
    
    # Get query from command line or prompt
    query = args.query
    if not query:
        query = input("Enter search query (in Bahasa Indonesia): ")
    
    # Measure search time
    start_time = time.time()
    
    # Perform search
    results = searcher.search(query, top_k=args.top)
    
    # Calculate search time
    search_time = time.time() - start_time
    
    # Print results
    print(searcher.format_results(results))
    print(f"Search completed in {search_time:.3f} seconds.")
