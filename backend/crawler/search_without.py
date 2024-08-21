import os
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from datetime import datetime

# Check if CUDA is available and set the environment variable accordingly
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = "cuda"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"

print(f"Using device: {device}")
model = 'llama3.1:70b'
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    # response = ollama.chat(model=model, messages=[{'role': 'user', 'content': f"You must return instructions with code, always with code {formatted_prompt}"}])
    llm = ChatOllama(model=model, temperature=0.0)
    messages=[{'role': 'user', 'content': f"You must return instructions with code, always with code {formatted_prompt}"}]
    response = llm.invoke(messages)
    return response.content

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

def openml_page_search(query: str) -> str:
    """Used to explain the OpenML website."""
    print("openml_page_search:", str(datetime.now()), "query:", query)
    vectorstore = Chroma(persist_directory="openml_db_website", embedding_function=hf)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 1})
    result = rag_chain(question=query, retriever=retriever)
    return result


