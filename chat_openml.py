from openml_backend import agent_response
import uuid

# Generate a unique session ID
session_id = str(uuid.uuid4())
print("this is the session id", session_id)
while True:
    print("########################################")
    print("#                                      #")
    print("#            Chat OpenML               #")
    print("#                                      #")
    print("########################################")
    print("#                                      #")
    print("#  Welcome to the Chat OpenML system!  #")
    print("#  Ask any question about datasets or  #")
    print("#  documentation. Type 'exit' to quit. #")
    print("#                                      #")
    print("########################################")
    # Ask a question about the webpage
    prompt = input("What would you like to know about OpenML? ")

    # Exit condition
    if prompt.lower() == 'exit':
        print("Exiting the chatbot. Goodbye!")
        break

    # Chat with the webpage
    if prompt:
        result = agent_response(id, prompt)
        print(result)
