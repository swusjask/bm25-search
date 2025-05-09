# BM25 PDF Search

A simple Python tool for searching PDF documents in Bahasa Indonesia using the BM25 algorithm. The system extracts text from PDFs, indexes them, and allows for keyword-based searching to find the most relevant documents.

## Requirements

- Python 3.8 or higher
- PyPDF2 (or other PDF extraction library)
- rank_bm25
- Additional dependencies as needed

## Setup

1. Clone this repository
2. Install required packages from requirements.txt:
   ```
   pip install -r requirements.txt
   ```

## Usage

The system consists of two main Python scripts:

### 1. Indexing (index.py)

Run this first to extract text from PDFs and create the index:

```
python index.py
```

This processes all PDF files from the `documents` folder and stores the index in the `index` folder.

### 2. Searching (search.py)

After indexing, run the search script in one of the following ways:

#### Basic usage (with prompt):
```
python search.py
```
This will prompt you to enter your search query in Bahasa Indonesia.

#### Using command-line arguments:
```
python search.py --query "pendidikan indonesia"
```

#### Additional options:
```
python search.py --query "teknologi informasi" --top 5
```

This example searches for "teknologi informasi", returns up to 5 results (default is 3)

#### Get help on available options:
```
python search.py --help
```

The search will display the most relevant documents with snippets showing where matches occurred.

## Structure

- `documents/`: Folder containing PDF documents to be searched
- `index/`: Folder where index data is stored
- `index.py`: Script for extracting text from PDFs and building the index
- `search.py`: Script implementing BM25 algorithm for searching indexed documents
