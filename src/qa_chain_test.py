from pathlib import Path

from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
VECTORSTORE_DIR = Path("data/vectorstore/faiss_index")

# Load the embedding model (used for retrieval)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index
print("Loading FAISS vectorstore...")
vector_store = FAISS.load_local(
        str(VECTORSTORE_DIR), 
        embeddings=embedding_model,
        allow_dangerous_deserialization = True
        )


# # Load FAISS index binary
# index = faiss.read_index(str(VECTORSTORE_DIR / "index.faiss"))

# # Load metadata + page_content
# docs = []
# docstore_dict = {}
# index_to_docstore_id = {}
# with open(VECTORSTORE_DIR / "metadata.jsonl", "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         entry = json.loads(line)
#         doc = Document(
#             page_content=entry["page_content"],
#             metadata=entry["metadata"]
#         )
#         doc_id = str(i)
#         docs.append(doc)
#         docstore_dict[doc_id] = doc
#         index_to_docstore_id[i] = doc_id

# docstore = InMemoryDocstore(docstore_dict)
# # Rebuild vectorstore
# vector_store = FAISS(embedding_model, index, docstore, index_to_docstore_id=index_to_docstore_id)

# Initialize LLM (Using Ollama-Mistral)
llm = OllamaLLM(model='mistral')

# Create retriever-QA chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever,
)
qa_chain.return_source_documents = True

# Main
print("\n Document QA is ready! Ask a question (or type 'exit'):\n")

while True:
    query = input(" Your question (or type exit to quit): ")
    if query.strip().lower() in ["exit","quit"]:
        break

    if not query:
            continue  # skip empty inputs

    ## Logging
    # Fetch docs to get the context text
    # retrieved_docs = retriever.invoke(query)
    # context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Access the prompt template used internally
    # prompt_template = qa_chain.combine_documents_chain.llm_chain.prompt

    # Format the prompt exactly as sent to the LLM
    # full_prompt = prompt_template.format(context=context_text, question=query)

    # print("\n Prompt sent to model:\n")
    # print(full_prompt)
    # print("\n" + "-"*60)
    ## End Logging

    result = qa_chain.invoke({'query':query})

    print("\n Answer: ",result['result'])
    

    # Print sources
    print("\n Source Documents Used:\n")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "N/A")
        preview = doc.page_content[:300].replace("\n", " ")  # first 300 chars, single line
        print(f"{i}. Source: {source}, Page: {page}")
        print(f"   Preview: {preview}...\n")


    print("\n" + "="*80)