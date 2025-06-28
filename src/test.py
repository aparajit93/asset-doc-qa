from langchain_core.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
# prompt = ChatPromptTemplate([
#     ('human',"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:")
# ])

# prompt = PromptTemplate(
#     input_variables=['context', 'question'],
#     template= "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. \nQuestion: {question} \nContext: {context} \nAnswer:"
# )

print(type(prompt))