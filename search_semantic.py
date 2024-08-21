from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
import requests
import pandas as pd
from session import get_session_history

metadata = pd.read_csv("/home/israel/AIoD/Talk2OpenML_v3/semantic_search/ai_search/data/all_dataset_description.csv")
rag_response_path = 'http://0.0.0.0:8000/' # modify depending on the server

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
