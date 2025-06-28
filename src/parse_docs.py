from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from file_browser import list_documents, select_documents

DOC_DIR = "data/documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_documents(paths: list[Path]) -> list:
    # Load supported documents from user-selected file paths
    docs = []
    for path in paths:
        if path.suffix.lower() == '.pdf':
            loader = PyMuPDFLoader(str(path))
        elif path.suffix.lower() == '.txt':
            loader = TextLoader(str(path))
        else:
            print(f" Unsupported file type: {path.name}")
            continue

        loaded = loader.load()
        print(f" Loaded {len(loaded)} pages from {path.name}")
        docs.extend(loaded)

    return docs

def chunk_documents(docs: list) -> list:
    # Split raw documents into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len
    )
    chunks = splitter.split_documents(docs)
    print(f' Split into {len(chunks)} chunks')
    return chunks

if __name__ == "__main__":
    files = list_documents(DOC_DIR)
    selected = select_documents(files)

    if selected is None:
        print(" Selection cancelled. Exiting.")
        exit(0)

    raw_docs = load_documents(selected)
    if not raw_docs:
        print(" No documents loaded. Exiting.")
        exit(0)

    chunks = chunk_documents(raw_docs)

    print(f' {len(chunks)} chunks ready for embedding.')