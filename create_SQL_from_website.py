
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# crawl_df = pd.read_csv("openml_docs.csv")
crawl_df = pd.read_csv("openml_docs_API_together.csv")

# Create a new column 'joined' by concatenating the contents of the specified columns
crawl_df['joined'] = crawl_df.apply(
    lambda row: f"url: {row['URL']}, body_text: {row['Body Text']}, header_links_text: {row['Header Links Text']}, h1: {row['H1']}, h2: {row['H2']}, h3: {row['H3']}, h4: {row['H4']}, title: {row['Title']}",
    axis=1
)

print('Columns of crawl_df', crawl_df.columns)
print('crawl_df', crawl_df)
print('This is the lenght of the joined column', len(crawl_df['joined']))
print('This is the joined column', crawl_df['joined'])
# Use the DataFrameLoader with the new 'joined' column
loader = DataFrameLoader(crawl_df, page_content_column="joined")
docs = loader.load()

char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0, separators=[" ", ",", "\n"])
doc_texts = char_text_splitter.split_documents(docs)

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

for doc in doc_texts:
    for md in doc.metadata:
        doc.metadata[md] = str(doc.metadata[md])
print("Creating the vector store")
Chroma.from_documents(documents=doc_texts, embedding=hf, persist_directory='openml_db_2024-08-13')
