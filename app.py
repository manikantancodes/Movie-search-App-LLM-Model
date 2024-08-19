# Import required libraries
import pandas as pd
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import openai
import streamlit as st
import time

# Set your OpenAI API key
api_key = 'your api key'
openai.api_key = api_key

# Function to load and prepare dataset
def load_and_prepare_data(url):
    df = pd.read_csv(url)
    
    # Handle missing values by replacing NaN with empty string
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['genres'] = df['genres'].fillna('')
    
    # Combine columns to form a single text column
    df['text'] = df['title'] + " " + df['description'] + " " + df['genres'].astype(str)
    
    # Join all text data into a single document
    doc = "\n".join(df['text'].tolist())
    return doc

# Example URL - replace with your actual CSV URL
url = 'https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv'
doc = load_and_prepare_data(url)
print(doc[:1000])  # Print first 1000 characters of the document to verify


# Function to split text into chunks
def split_text(doc):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    texts = splitter.split_text(doc)
    return texts

# Split the document
texts = split_text(doc)
print(f'Number of chunks: {len(texts)}')
print(texts[:2])  # Print the first two chunks to verify


# Function to prepare the vector store
def prepare_vector_store(text, api_key):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    texts = splitter.split_text(text)
    embed_text = FAISS.from_texts(texts, OpenAIEmbeddings(openai_api_key=api_key))
    return embed_text

# Load QA chain
def load_qa_chain(api_key):
    model = load_qa_chain(OpenAI(openai_api_key=api_key), chain_type="stuff")
    return model


def handle_rate_limit():
    st.error("You have exceeded your API quota. Please try again later.")
    time.sleep(10)  # Wait for a while before retrying


# Streamlit application

def main():
    st.title("Movie QA Application")
    st.write("Welcome to the Movie QA application! Ask any question about movies, and get answers based on our dataset.")
    
    # Define the URL for the movie dataset
    url = 'https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv'
    
    # Load and prepare data
    doc = load_and_prepare_data(url)
    
    # Prepare vector store
    embed_text = prepare_vector_store(doc, openai.api_key)
    
    # Load QA chain
    model = load_qa_chain(openai.api_key)
    
    # User input for query with custom styling
    query = st.text_input("Enter your question about movies:", placeholder="e.g., What is the highest-grossing movie of all time?")
    
    # Add a submit button
    if st.button("Submit"):
        if query:
            # Search for relevant documents
            my_docs = embed_text.similarity_search(query)
            
            # Generate an answer
            answer = model.run(input_documents=my_docs, question=query)
            
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question before clicking Submit.")
    
    # Add a footer with some information
    st.markdown("""
        <footer style="text-align: center; padding: 10px; background-color: #f5f5f5; border-top: 1px solid #ddd;">
            <p>Powered by Streamlit and OpenAI</p>
        </footer>
    """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()