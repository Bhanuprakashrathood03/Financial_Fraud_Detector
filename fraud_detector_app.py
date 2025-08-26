# -*- coding: utf-8 -*-
"""
Financial Fraud Detection using LLM and RAG

This script leverages Large Language Models (LLM) and Retrieval-Augmented 
Generation (RAG) to identify potential fraud in financial data.
"""

import os
import re
import random
import warnings
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
# Set environment variable for CUDA to prevent potential issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Define the model ID for the Hugging Face model
MODEL_ID = 'HuggingFaceH4/zephyr-7b-beta'
# Define the directory to persist the Chroma DB
PERSIST_DIRECTORY = 'docs/chroma_rag/'
# Define the collection name for Chroma DB
COLLECTION_NAME = "finance_data_new"


def generate_financial_data():
    """
    Generates a synthetic dataset of fraudulent and non-fraudulent
    financial statements and saves it to a CSV file.
    """
    print("Generating synthetic financial data...")
    fraud_statements = [
        "The company reported inflated revenues by including sales that never occurred.",
        "Financial records were manipulated to hide the true state of expenses.",
        "The company failed to report significant liabilities on its balance sheet.",
        "Revenue was recognized prematurely before the actual sales occurred.",
        "The financial statement shows significant discrepancies in inventory records.",
        "The company used off-balance-sheet entities to hide debt.",
        "Expenses were understated by capitalizing them as assets.",
        "There were unauthorized transactions recorded in the financial books.",
        "Significant amounts of revenue were recognized without proper documentation.",
        "The company falsified financial documents to secure a larger loan.",
        "There were multiple instances of duplicate payments recorded as expenses.",
        "The company reported non-existent assets to enhance its financial position.",
        "Expenses were fraudulently categorized as business development costs.",
        "The company manipulated financial ratios to meet loan covenants.",
        "Significant related-party transactions were not disclosed.",
        "The financial statement shows fabricated sales transactions.",
        "There was intentional misstatement of cash flow records.",
        "The company inflated the value of its assets to attract investors.",
        "Revenue from future periods was reported in the current period.",
        "The company engaged in channel stuffing to inflate sales figures."
    ]

    non_fraud_statements = [
        "The company reported stable revenues consistent with historical trends.",
        "Financial records accurately reflect all expenses and liabilities.",
        "The balance sheet provides a true and fair view of the company’s financial position.",
        "Revenue was recognized in accordance with standard accounting practices.",
        "The inventory records are accurate and match physical counts.",
        "The company’s debt is fully disclosed on the balance sheet.",
        "All expenses are properly categorized and recorded.",
        "Transactions recorded in the financial books are authorized and documented.",
        "Revenue recognition is supported by proper documentation.",
        "Financial documents were audited and found to be accurate.",
        "Payments and expenses are recorded accurately without discrepancies.",
        "The assets reported on the balance sheet are verified and exist.",
        "Business development costs are properly recorded as expenses.",
        "Financial ratios are calculated based on accurate data.",
        "All related-party transactions are fully disclosed.",
        "Sales transactions are accurately recorded in the financial statement.",
        "Cash flow records are accurate and reflect actual cash movements.",
        "The value of assets is fairly reported in the financial statements.",
        "Revenue is reported in the correct accounting periods.",
        "Sales figures are accurately reported without manipulation."
    ]

    fraud_data = [{"text": statement, "fraud_status": "fraud"} for statement in fraud_statements]
    non_fraud_data = [{"text": random.choice(non_fraud_statements), "fraud_status": "non-fraud"} for _ in range(60)]

    data = fraud_data + non_fraud_data
    random.shuffle(data)

    df = pd.DataFrame(data)
    df.to_csv("financial_statements_fraud_dataset.csv", index=False)
    print("Dataset 'financial_statements_fraud_dataset.csv' created successfully.")
    return df

def clean_text_data(df):
    """
    Cleans the text data in the DataFrame by removing punctuation, numbers,
    stopwords, and converting to lowercase.
    """
    print("Cleaning text data...")
    # Ensure NLTK resources are downloaded
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    def clean_text(text):
        text = text.encode('ascii', 'ignore').decode()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['Clean_Text'] = df['text'].apply(clean_text)
    df.drop(columns=['text'], inplace=True)
    print("Text data cleaned.")
    return df

def create_vector_store(df):
    """
    Creates a Chroma vector store from the cleaned financial data.
    """
    print("Creating Chroma vector store...")
    documents = []
    for i, row in df.iterrows():
        document_content = f"id:{i}\\Fillings: {row['Clean_Text']}\\Fraud_Status: {row['fraud_status']}"
        documents.append(Document(page_content=document_content))

    hg_embeddings = HuggingFaceEmbeddings()

    langchain_chroma = Chroma.from_documents(
        documents=documents,
        collection_name=COLLECTION_NAME,
        embedding=hg_embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("Vector store created and persisted.")
    return langchain_chroma

def initialize_llm_pipeline():
    """
    Initializes the Hugging Face LLM and the text generation pipeline.
    """
    print("Initializing the LLM and text generation pipeline...")
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set quantization configuration
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, max_new_tokens=1024)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Initialize the query pipeline
    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=6000,
        max_new_tokens=500,
        device_map="auto",
    )
    
    llm = HuggingFacePipeline(pipeline=query_pipeline)
    print("LLM pipeline initialized.")
    return llm

def create_qa_chain(llm, vector_store):
    """
    Creates the RetrievalQA chain with a custom prompt.
    """
    print("Creating the QA chain...")
    template = """
You are an expert financial analyst specializing in fraud detection. Your task is to assess a given financial statement and classify its risk of fraud.
Based *only* on the provided Context document, classify the Question into one of the following categories:
- High Risk of Fraud
- Low Risk of Fraud
- Indeterminate Risk

After the classification, provide a brief, one-sentence justification for your decision, citing the context.
If the context is not relevant to the question, you must classify it as 'Indeterminate Risk'.

Question: {question}
Context: {context}

Classification:
Justification:
"""
    PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever, 
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    print("QA chain created.")
    return qa_chain

def main():
    """
    Main function to run the fraud detection process.
    """
    # Step 1: Generate or load data
    if os.path.exists("financial_statements_fraud_dataset.csv"):
        print("Loading existing dataset...")
        df = pd.read_csv("financial_statements_fraud_dataset.csv")
    else:
        df = generate_financial_data()

    # Step 2: Clean the text data
    df_cleaned = clean_text_data(df.copy())
    
    # Step 3: Create or load the vector store
    if os.path.exists(PERSIST_DIRECTORY):
        print("Loading existing vector store...")
        hg_embeddings = HuggingFaceEmbeddings()
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=hg_embeddings, collection_name=COLLECTION_NAME)
    else:
        vector_store = create_vector_store(df_cleaned)

    # Step 4: Initialize the LLM
    llm = initialize_llm_pipeline()

    # Step 5: Create the QA chain
    qa_chain = create_qa_chain(llm, vector_store)

    # Step 6: Ask questions and get predictions
    print("\n--- Fraud Detection System Ready ---")
    print("Enter a financial statement to analyze (or type 'exit' to quit).")

    while True:
        question = input("\nStatement: ")
        if question.lower() == 'exit':
            break
        
        try:
            result = qa_chain({"query": question})
            print("\n--- Analysis Result ---")
            print(f"Query: {result['query']}")
            print(f"\nResult: {result['result']}")
            print(f"\nSource Document: {result['source_documents'][0].page_content}")
            print("-----------------------\n")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()