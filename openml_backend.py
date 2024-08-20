from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import torch
import requests
import pandas as pd
from datetime import datetime

print("Backend started")

# Check if CUDA is available and set the environment variable accordingly
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = "cuda"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"

print(f"Using device: {device}")
### Statefully manage chat history ###
store = {}

metadata = pd.read_csv("./semantic_search/ai_search/data/all_dataset_description.csv")
model = 'llama3.1:70b'
rag_response_path = 'http://0.0.0.0:8000/' # modify depending on the server
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

llm = ChatOllama(model=model, temperature=0.0)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # print("this is the session id", session_id)
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # print("this is the store id", store[session_id])
    return store[session_id]
    
def openml_page_search(id: str, input: str) -> str:
    system_prompt = (
        "You are an assistant for question-answering tasks related to OpenML. "
        "Use the following pieces of retrieved context to answer "
        "the question. You must return instructions with code, always with code"
        "\n\n"
        "{context}"
    )

    vectorstore = Chroma(persist_directory="openml_db_website", embedding_function=hf)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 1})
    
    
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history about OpenML, "
        "formulate a standalone question which can be understood "
        "without the chat history. If the question can be answer withou previous context"
        "also reformulate the question. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks from OpenML. "
        "Use the following pieces of retrieved context to answer "
        "the question. You must return instructions with code, always with code" 
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    answer = conversational_rag_chain.invoke(
        {"input": f"{input}"},
        config={
            "configurable": {"session_id": id}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
    return answer

def fetch_rag_response(query, query_type='dataset'):
        """
        Description: Fetch the response from RAG pipeline

        """
        element = requests.get(
            f"{rag_response_path}{query_type.lower()}/{query}",
            json={"query": query, "type": query_type.lower()},
        )
        rag_response = element.json()
        return rag_response

def database_similarity_search(id: str, query: str) -> str:
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
    # Reset index to remove it from the DataFrame
    filtered_metadata.reset_index(drop=True, inplace=True)
    # Update session history
    session_history = get_session_history(session_id=id)
    user_message = HumanMessage(content=query)
    session_history.add_user_message(user_message)
    assistant_message = AIMessage(content=filtered_metadata.to_string())
    session_history.add_ai_message(assistant_message)
    # session_history.add_message([user_message, assistant_message])
    
    return filtered_metadata

def agent_response(id: str, input_query: str) -> str:
    """
    Determines if the user is asking for 'code' or 'IDs' using the LLM and calls the appropriate function.
    :param input_query: The user's query
    :return: The result from the appropriate search function
    """
    # Use the LLM to interpret the input query
    # Conver store[session_id] to a string
    store_str = str(store)
    messages=[{'role': 'user', 'content': f"Interpret the user's motive, if you identify that the user is talking about code or wants to know how to do something in python, you should return the word 'code', if you identify that the user wants the id of a dataset or to find a dataset through keywords, you should return the word 'id'. You must return only one word, 'code' or 'id' (always in lower case), consider the previous context to make your decision, this is the context: {store_str + '. This is the last request from the user: ' +  input_query}"}]
    response = llm.invoke(messages)
    interpretation = response.content


    print("This is the interpretation:", interpretation)
    if 'code' in interpretation:
        print("LLM identified request for code.")
        return openml_page_search(id, input_query)
    elif 'id' in interpretation:
        print("LLM identified request for IDs.")
        return database_similarity_search(id, input_query)
    else:
        print("LLM interpretation did not match 'code' or 'id'.")
        return "LLM interpretation did not match 'code' or 'id'. Please specify your request clearly."
