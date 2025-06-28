import json
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def append_metadata(chunks, metadata_path):
    with open(metadata_path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }, f)
            f.write("\n")

def load_documents(paths: list[Path]) -> list:
    # Load supported documents from user-selected file paths
    docs = []
    for path in paths:
        if path.suffix.lower() == '.pdf':
            loader = PyMuPDFLoader(str(path))
        else:
            print(f" Unsupported file type: {path.name}")
            continue

        loaded = loader.load()
        #print(f" Loaded {len(loaded)} pages from {path.name}")
        docs.extend(loaded)

    return docs

def chunk_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    # Split raw documents into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    chunks = splitter.split_documents(docs)
    # print(f' Split into {len(chunks)} chunks')
    return chunks

