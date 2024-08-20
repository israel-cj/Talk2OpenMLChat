
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from datetime import datetime
import os
import ollama
import requests
import pandas as pd
import torch

print("Backend started")

# Check if CUDA is available and set the environment variable accordingly
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = "cuda"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"

print(f"Using device: {device}")

metadata = pd.read_csv("./semantic_search/ai_search/data/all_dataset_description.csv")
model = 'llama3.1:70b'
rag_response_path = 'http://0.0.0.0:8000/' # modify depending on the server
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

def fetch_rag_response(query, query_type='dataset'):
        """
        Description: Fetch the response from RAG pipeline

        """
        rag_response = requests.get(
            f"{rag_response_path}{query_type.lower()}/{query}",
            json={"query": query, "type": query_type.lower()},
        ).json()
        return rag_response

def database_similarity_search(query: str) -> str:
    """
    Used to search for resources a user needs.
    :param query: user input
    :param session_id: user session id
    :return:
    """
    print(str(datetime.now()), "database_similarity_search_v3", "query:", query)
    result = fetch_rag_response(query, query_type='dataset')
    filtered_metadata = metadata[
                        metadata["did"].isin(result['initial_response'])
                    ]
    # Return columns did, name, NumberOfInstances, NumberOfFeatures
    filtered_metadata = filtered_metadata[["did", "name", "NumberOfInstances", "NumberOfFeatures"]]
    # filtered_metadata = filtered_metadata[["did", "name", "description"]]
    return filtered_metadata


def agent_response(input_query: str) -> str:
    """
    Determines if the user is asking for 'code' or 'IDs' using the LLM and calls the appropriate function.
    :param input_query: The user's query
    :return: The result from the appropriate search function
    """
    # Use the LLM to interpret the input query
    response_user = ollama.chat(model=model, messages=[{'role': 'user', 'content': f"Interpret the user's motive, if you identify that the user is talking about code or wants to know how to do something in python, you should return the word 'code', if you identify that the user wants the id of a dataset or to find a dataset through keywords, you should return the word 'id', this is the request. You must return only one word, 'code' or 'id' (always in lower case): {input_query}"}])
    interpretation = response_user['message']['content']

    print("This is the interpretation:", interpretation)
    if 'code' in interpretation:
        print("LLM identified request for code.")
        return openml_page_search(input_query)
    elif 'id' in interpretation:
        print("LLM identified request for IDs.")
        return database_similarity_search(input_query)
    else:
        print("LLM interpretation did not match 'code' or 'id'.")
        return "LLM interpretation did not match 'code' or 'id'. Please specify your request clearly."


