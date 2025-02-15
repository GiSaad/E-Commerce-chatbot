#Original file is located at
#   https://colab.research.google.com/drive/1EmLuWV-69NGpdnwXQn_0t541VrbkVsmo

# **Amazon Product Chatbot with Mistral-7B and LangChain**

# === Environment Setup ===
# Install required packages with version pinning for reproducibility

!pip install transformers langchain sentence-transformers chromadb accelerate bitsandbytes

!pip install --upgrade --no-cache-dir bitsandbytes

!pip install --upgrade transformers accelerate bitsandbytes



from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HUGGINGFACE_TOKEN = "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"

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN,
                                             quantization_config=quantization_config,
                                             device_map="auto")

from transformers import pipeline

chatbot = pipeline(
    "text-generation",  # Use "text-generation" or "conversational" depending on the model
    model=model,
    tokenizer=tokenizer,
    device_map="auto",  # Automatically assigns the model to GPU if available
    max_new_tokens=200  # Controls the response length
)

response = chatbot("Tell me a fun fact about space.")
print(response[0]["generated_text"])

!pip install -U langchain-community

import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

df = pd.read_csv("/content/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv", on_bad_lines="skip")

print(df.columns)

!pip install chromadb

# Convert product data into retrievable format
documents = [Document(page_content=f"{row['Product Name']} - {row['About Product']}- stock : {row['Stock']} - Price: {row['Selling Price']} - product_id: {row['Uniq Id']}- category: {row['Category']}")
             for _, row in df.iterrows()]

# Use sentence transformers for embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Store product data in ChromaDB
db = Chroma.from_documents(documents, embedding_function)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20  # Number of docs to consider for MMR
    }
)

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline


# Ensure LLM is correctly wrapped
llm = HuggingFacePipeline(pipeline=chatbot)

# Create RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Default method
    retriever=retriever,
    return_source_documents=False  # Ensures only answers are returned

)

query = "What are the  Products under 30  dollars?"
result = qa.run(query)
print(result)

!pip install gradio

!pip install -U gradio

import gradio as gr

def chat_with_bot(user_input):
    print(f"User input: {user_input}")  # Debugging
    try:
        response = qa.run(user_input)
        response = response.split("Helpful Answer:")[-1].strip()  # Extract answer

        print(f"Bot response: {response}")  # Debugging
        return response
    except Exception as e:
        print(f"Error: {e}")
        return "Error processing your request."

demo = gr.Interface(fn=chat_with_bot, inputs="text", outputs="text")
demo.launch()
