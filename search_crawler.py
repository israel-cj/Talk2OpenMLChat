import os
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Assuming 'crawler' and 'history' are part of the same package
from session import get_session_history

print("Backend started")

# Check if CUDA is available and set the environment variable accordingly
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = "cuda"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"


model = 'llama3.1:70b'
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
llm = ChatOllama(model=model, temperature=0.0)

    
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
