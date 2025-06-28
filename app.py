import os
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils import load_documents, chunk_documents
import tempfile


# --- GLOBAL ---
VECTORSTORE_DIR = Path("data/vectorstore/faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
#llm = OllamaLLM(model='mistral')
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Cache ---
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore():
    embedding_model = load_embedder()
    if (VECTORSTORE_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(VECTORSTORE_DIR), 
            embeddings=embedding_model, 
            allow_dangerous_deserialization = True
            )
    else:
        return None

def load_model():
    return OllamaLLM(model='mistral')

# --- CONFIG ---
st.set_page_config(page_title="Asset Document QA", layout="wide")
st.title("Ask Your Asset Docs")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload Section ---
with st.form('upload_form'):
    uploaded_files = st.file_uploader(
        label= "Upload documents",
        type = 'pdf',
        accept_multiple_files= True
    )
    submit = st.form_submit_button("Update documents")

if submit and uploaded_files:
    with st.spinner(text= "Processing uploaded documents"):
        temp_paths = []
        try:
            all_docs = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.getbuffer())
                    temp_paths.append(Path(tmp.name))
                    loader = PyMuPDFLoader(tmp.name)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = file.name
                    all_docs.extend(docs)
            chunks = chunk_documents(all_docs, CHUNK_SIZE, CHUNK_OVERLAP)

            embedding_model = load_embedder()
            if 'vector_store' in st.session_state:
                st.session_state.vector_store.add_documents(chunks)
            else:
                st.session_state.vector_store = FAISS.from_documents(chunks, embedding_model)

            st.session_state.vector_store.save_local(str(VECTORSTORE_DIR))
        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    st.warning(f"Could not delete temp file: {path} ({e})")
    st.success("Documents Processed")

# --- Query Section ---
st.markdown("---")
if 'vector_store' not in st.session_state:
    vs = get_vectorstore()
    if vs:
        st.session_state.vector_store = vs

if 'qa_chain' not in st.session_state and 'vector_store' in st.session_state:
    llm = load_model()
    prompt_template = PromptTemplate(
        input_variables=['input', 'context'],
        template= "You are an assistant for question-answering tasks." \
        " Use the following pieces of retrieved context to answer the question. " \
        "If you don't know the answer, just say that you don't know, don't try to make up an answer." \
        "\nQuestion: {input} \nContext: {context} \nAnswer:"
    )

    prompt = ChatPromptTemplate(input_variables = ['input','context'], messages=[HumanMessagePromptTemplate(prompt=prompt_template)])

    retriever = st.session_state.vector_store.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

query = st.chat_input("Ask about the documents...")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query and 'qa_chain' in st.session_state:

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = st.session_state.qa_chain.invoke({'input':query})
            answer = result['answer']
            sources = result['context']

            st.markdown("Answer")
            st.markdown(f"{answer}")
            st.markdown("---")    

            # Print sources
            st.markdown("Source Documents Used")
            for i, doc in enumerate(sources, 1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page", "N/A")
                preview = doc.page_content[:300].replace("\n", " ")
                st.markdown(f"**{i}. Source:** `{source}` — **Page:** {page}\n\n> {preview}...")
            
            
            if sources:
                source_texts = "\n\n".join(
                    [f"**{i}. Source:** `{doc.metadata.get('source', 'unknown')}` — Page {doc.metadata.get('page', 'N/A')}\n> {doc.page_content[:300].replace(chr(10), ' ')}..." for i, doc in enumerate(sources, 1)]
                )
                answer += f"\n\n---\n📄 **Sources:**\n{source_texts}"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    if query:
        st.warning("No documents found. Please upload documents first.")
