from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # print("this is the session id", session_id)
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # print("this is the store id", store[session_id])
    return store[session_id]
