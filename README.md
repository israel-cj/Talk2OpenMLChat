# Talk2OpenMLChat

# Talk2OpenMLChat

## Introduction
Talk2OpenMLChat is a chatbot designed to interact with [OpenML](https://www.openml.org/). It allows users to query and retrieve information from OpenML in a conversational manner.

## Installation

### Prerequisites
- Python 3.11 (recommended)


## Note on without
The chat_openml_without.py script runs the chatbot without keeping track of the conversation history. This means each query is treated independently, without context from previous interactions. On the other hand chat_openml.py keeps the history.

### Steps

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/Talk2OpenMLChat.git
    cd Talk2OpenMLChat
    ```

2. **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Install Ollama and download the models:**
    - Visit [Ollama](https://ollama.com/) for installation instructions.
    - Download the required models:
        ```sh
        ollama pull llama3.1:70b
        ```

4. **Create the CSV file for the crawler:**
    - If you want to skip and go directly to step 6, you can download the SQL database from [this link](https://tuenl-my.sharepoint.com/:u:/g/personal/i_campero_jurado_tue_nl/EefZPL9EcV9Iukvs3dPBsn8BlQxBTFhHW4qPOwqMJYwRVg?e=bMANbq). Otherwise, the CSV file (`openml_docs_API_together.csv`) needs to be created by crawling the following URLs and their sublinks:
        - "https://openml.github.io/openml-python/main/"
        - "https://docs.openml.org/"
    - Run the script to create the CSV:
        ```sh
        python create_csv_for_crawler.py
        ```

5. **Embed the CSV into a SQL database:**
    - Execute the following script to create the SQL database inside `openml_db_website`:
        ```sh
        python create_SQL_from_website.py
        ```
    

6. **Run the backend for semantic search:**
    - Follow the instructions from the [ai_search repository](https://github.com/openml-labs/ai_search/tree/main) to set up the backend.

## Configuration

Once the backend from `ai_search` is working, you need to modify two lines in `openml_backend.py` or `openml_backend_without.py` depending on the type of chatbot you want to run:

```python
metadata = pd.read_csv("./semantic_search/ai_search/data/all_dataset_description.csv")
rag_response_path = 'http://0.0.0.0:8000/'
```

Replace these lines with the appropriate paths and port numbers based on your setup.


To run the chatbot with conversation history tracking:
```python
python chat_openml.py
```
To run the chatbot without conversation history tracking:
```python
python chat_openml_without.py
```

Enjoy using Talk2OpenMLChat!