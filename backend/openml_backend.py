from langchain_ollama import ChatOllama
from crawler.search import openml_page_search
from backend.semantic_search.search import database_similarity_search
from backend.history.session import get_session_history
print("Backend started")

model = 'llama3.1:70b'
llm = ChatOllama(model=model, temperature=0.0)


def agent_response(id: str, input_query: str) -> str:
    """
    Determines if the user is asking for 'code' or 'IDs' using the LLM and calls the appropriate function.
    :param input_query: The user's query
    :return: The result from the appropriate search function
    """
    # Use the LLM to interpret the input query
    # Conver store[session_id] to a string
    store = get_session_history(id)
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
