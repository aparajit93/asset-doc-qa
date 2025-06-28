from pathlib import Path
import glob
import argparse
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from parse_docs import load_documents, chunk_documents
import faiss
import os

# === CONFIG ===
VECTORSTORE_DIR = Path("data/vectorstore")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def update_documents(file_paths: list[str]):
    docs = []
    for path in file_paths:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(str(path))
        elif path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(path))
        else:
            print(f"Unsupported file format: {path.name}")
            continue
        loaded = loader.load()
        print(f" Loaded {len(loaded)} pages from {path.name}")
        docs.extend(loaded)
    return docs

def update_vectorstore(new_chunks, vectorstore_dir, model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    print("Updating vectorstore")
    if (vectorstore_dir / "index.faiss").exists():
        # Load existing index
        vectorstore = FAISS.load_local(
            str(vectorstore_dir), 
            embeddings=embedding_model,
            allow_dangerous_deserialization = True
        )
        vectorstore.add_documents(new_chunks)
    else:
        # Create new index
        vectorstore = FAISS.from_documents(new_chunks, embedding=embedding_model)

    # Save updated vectorstore
    vectorstore.save_local(str(vectorstore_dir))
    print("Vectorstore updated.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Upload and embed new documents.")
    parser.add_argument("files", nargs="+", help="Paths to files to upload")
    args = parser.parse_args()
    expanded_paths = []
    for pattern in args.files:
        expanded_paths.extend(glob.glob(pattern))

    docs = update_documents(expanded_paths)
    if not docs:
        print("No valid documents loaded.")
        exit()

    chunks = chunk_documents(docs)
    update_vectorstore(chunks, VECTORSTORE_DIR, EMBEDDING_MODEL_NAME)
