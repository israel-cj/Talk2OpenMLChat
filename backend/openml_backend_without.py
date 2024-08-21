import ollama
from crawler.search_without import openml_page_search
from semantic_search.search_without import database_similarity_search

model = 'llama3.1:70b'

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


