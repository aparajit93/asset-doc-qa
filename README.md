# 🛠️ Project: Asset Document QA with Local LLMs

A private, local-first question-answering system for industrial asset documents — powered by LangChain, FAISS, and Ollama. Upload equipment manuals, datasheets, or schematics, and query them conversationally using a chat interface.

No cloud APIs required.

---

## 🚀 Features

- 🧠 **LLM-based Question Answering**: Powered by [Ollama](https://ollama.com) and open-weight models like Mistral or LLaMA.
- 📄 **Multi-file Upload**: Supports uploading multiple PDFs for ingestion.
- 🧾 **Semantic Search**: Uses vector embeddings (FAISS) to retrieve relevant document chunks.
- 💬 **Chat UI with Context**: Streamlit chat interface with answer history and source tracing.
- 🔐 **Local + Private**: Runs entirely offline with no calls to OpenAI or external APIs.

---

## 📦 Tech Stack

| Component      | Tool/Lib              |
|----------------|------------------------|
| LLM Inference  | [Ollama](https://ollama.com) (`mistral`, `llama3`, etc.) |
| RAG Pipeline   | [LangChain](https://github.com/langchain-ai/langchain) |
| Embeddings     | HuggingFace or Ollama-supported models |
| Vector DB      | FAISS                 |
| UI             | Streamlit             |
| File Parsing   | PyMuPDF / LangChain Loaders |

---

```bash
asset-doc-qa/
├── data/
│   ├── raw_pdfs/              # Source manuals, SOPs, etc.
│   └── processed/             # Cleaned chunks or metadata
│   └── vectorstore/           # FAISS vectorestore location
├── src/
│   ├── ingest.py              # PDF parsing and chunking
│   ├── embed_store.py         # Embedding + vector DB logic
│   ├── qa_chain.py            # LangChain QA pipeline
│   └── file_browser.py        # File browsing
│   └── parse_docs.py          # Alt PDF parsing and chunking
│   └── update_store.py        # Alt Embedding + vector DB logic
│   └── utils.py               # Utility functions
├── app.py                     # Streamlit UI
├── app_test.py                # Streamlit UI testing
├── requirements.txt
└── README.md
```

---

## 🧪 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/aparajit93/asset-doc-qa.git
cd asset-doc-qa
```
### 2. Set up Environment
```bash
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
### 3. Install Ollama and Mistral
Install Ollama and run
```bash
ollama pull mistral
```
### 4. Create Directory for Vectorstore (if it doesn't exist)
Install Ollama and run
```bash
mkdir data/vectorstore/faiss_index
```
### 5. Run the app
```bash
streamlit run app.py
```