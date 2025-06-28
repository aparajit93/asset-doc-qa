# Import Libraries

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import pickle

# Paths
RAW_DOCS_DIR = Path('data/raw_pdfs')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents= True, exist_ok= True)

# Load Documents
def load_documents():
    all_docs = []
    for pdf_path in RAW_DOCS_DIR.glob('*.pdf'):
        print(f'Loading {pdf_path.name}')
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

# Create chunks
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len
    )
    print("Splitting documents into chunks")
    return splitter.split_documents(docs)

# Save chunks
def save_chunks(chunks):
    out_path = PROCESSED_DIR/'chunks.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {out_path}")

# Main
def main():
    print("Loading Documents")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    print("Chunking Documents")
    chunks = chunk_documents(docs)
    print(f"Generated {len(chunks)} chunks")

    save_chunks(chunks)

if __name__ == '__main__':
    main()