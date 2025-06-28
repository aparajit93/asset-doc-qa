# Import Libraries
import pickle
import json
from pathlib import Path
import faiss
from langchain_community.vectorstores import FAISS # Import FAISS for storing vectors
from langchain_huggingface import HuggingFaceEmbeddings # Library to get models for embedding vectors

# Paths
PROCESSED_DIR = Path("data/processed")
VECTORSTORE_DIR = Path("data/vectorstore")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_PATH = PROCESSED_DIR / "chunks.pkl"
INDEX_PATH = Path("data/vectorstore/faiss_index")
INDEX_PATH.mkdir(parents=True, exist_ok=True)

# Load chunks
print("Loading chunks")
with open(CHUNKS_PATH, 'rb') as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")

# Initialize embedding model
print("Initializing embedding model")
embedding_model = HuggingFaceEmbeddings(model='all-MiniLM-L6-V2')

# Embedd and store in FAISS
vector_store = FAISS.from_documents(chunks, embedding_model)

# Save FAISS index
vector_store.save_local(str(INDEX_PATH))

# Save FAISS index binary
#faiss.write_index(vector_store.index, str(INDEX_PATH / "index.faiss"))

# Save metadata + page_content safely as JSONL
# with open(INDEX_PATH / "metadata.jsonl", "w", encoding="utf-8") as f:
#     for doc in vector_store.docstore._dict.values():
#         json.dump({
#             "metadata": doc.metadata,
#             "page_content": doc.page_content
#         }, f)
#         f.write("\n")

print(f"FAISS index and metadata saved safely to {INDEX_PATH}")

#print(f"FAISS index saved to {INDEX_PATH}")