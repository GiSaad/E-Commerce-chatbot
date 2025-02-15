# E-Commerce-chatbot
# Amazon Product Chatbot with Mistral 7B, LangChain, and ChromaDB

  

*This project implements a retrieval-augmented chatbot using Mistral 7B, LangChain, and ChromaDB to query an Amazon product database. The chatbot answers product-related queries efficiently by leveraging retrieval-based question-answering (RAG).*

  

## Features

  

✅ **Uses Mistral 7B as the language model**

✅ **Stores product data in ChromaDB for efficient retrieval**

✅ **Implements LangChain's RetrievalQA for querying**

✅ **Supports quantized inference with bitsandbytes to optimize memory usage**

✅ **Runs on Google Colab**

  

## Installation

  

Install the required dependencies:

  
```python
!pip install transformers langchain sentence-transformers chromadb accelerate bitsandbytes

!pip install -U langchain-community

!pip install -U gradio

  

Log in to Hugging Face (replace with your token):

  

!huggingface-cli login

  ```

## Model Setup

  

Import necessary libraries and load Mistral 7B with quantization:

  ```python

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

  

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

  

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name,

quantization_config=quantization_config,

device_map="auto")

  ```

**Load and Process Amazon Dataset**
  
Download and load the Amazon Product Dataset from Kaggle:

  ```python

import pandas as pd

  

df = pd.read_csv("/content/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv", on_bad_lines="skip")

  ```

**Convert product data into retrievable format:**

  
```python
from langchain.schema import Document

  

documents = [

Document(page_content=f"{row['Product Name']} - {row['About Product']} - Price: {row['Selling Price']} - product_id: {row['Uniq Id']} - category: {row['Category']}")

for _, row in df.iterrows()

]

  
```
**Store and Retrieve Data using ChromaDB**

  
```python
from langchain.vectorstores import Chroma

from langchain.embeddings import SentenceTransformerEmbeddings

  

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(documents, embedding_function)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})
```
  

## Implement Retrieval-Based Chatbot

  
```python
from langchain.chains import RetrievalQA

from langchain.llms import HuggingFacePipeline

from transformers import pipeline

  

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=200)

llm = HuggingFacePipeline(pipeline=chatbot)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

  ```

## Test the chatbot:

  
```python
query = "What are the products under 30 dollars?"

result = qa.run(query)

print(result)

  ```

## Deploy with Gradio

  

Run the chatbot interface using Gradio, displaying only the helpful answer:
```python
  

import gradio as gr

  

def chat_with_bot(user_input):

try:

response = qa.run(user_input)

response = response.split("Helpful Answer:")[-1].strip() # Extract only the helpful answer

return response

except Exception as e:

return f"Error: {e}"

  

demo = gr.Interface(fn=chat_with_bot, inputs="text", outputs="text")

demo.launch()

  ```

## Notes

  

Colab users: If running out of memory, try restarting the runtime and using a smaller batch size.

  

Hugging Face model: Ensure that you have access to the model and that your token is correctly configured.

  

## Future Improvements

  

🔹 Optimize retrieval strategy with better embeddings

🔹 Add Flask/FastAPI for API deployment

🔹 Improve response formatting

  

## License

  

This project is open-source. Feel free to modify and extend it!
