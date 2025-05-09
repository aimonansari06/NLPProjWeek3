# 🧠 Document Retriever with FAISS and Sentence Transformers

This project provides a lightweight document retrieval system powered by [Sentence-Transformers](https://www.sbert.net/) and [FAISS](https://github.com/facebookresearch/faiss). It supports searching across `.txt`, `.md`, and `.pdf` documents using semantic similarity.

---

## 📁 Project Structure
├── Week3
  ── Task3.ipynb # Jupyter notebook for interactive use
├──retriever
  ──retriever.py # Core Retriever class
├── testfile
  ── create_test_files.py # Script to generate example documents
├── test_retriever.py # Script to test retriever functionality
├── doc1.txt # Sample text file
├── doc2.md # Sample markdown file
├── README.md # You're here!

## 🔧 Installation

Make sure you have Python 3.7+ and install dependencies using pip:

```bash
%pip install openai sentence-transformers faiss-cpu hf_xet

## 🧪 Features
✅ Supports .txt, .md we tried to do pdf but it doesn't work due to package error

✅ Fast similarity search with FAISS

✅ Metadata tracking

✅ Persistent save/load of index
