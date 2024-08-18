from dotenv import load_dotenv
import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import logging

def upload_files():
    uploaded_files = st.file_uploader("Upload the PDF files", accept_multiple_files=True)
    return uploaded_files

def main():
    # Load environment variables from a .env file if it exists
    load_dotenv()

    # Retrieve the OpenAI API key from the environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if openai.api_key is None:
        st.error("OpenAI API key not found. Please set it in the environment variable 'OPENAI_API_KEY'.")
        return

    # Load a pre-trained OpenAI language model
    llm = OpenAI(model="gpt-3.5-turbo-instruct")

    # Configure the page settings for the Streamlit app
    st.set_page_config(page_title="Chat with PDF")

    # Display the header for the Streamlit app
    st.header("LangChain RAG App")

    # Allow users to upload PDF files
    pdfs = upload_files()

    # Check if PDF files have been uploaded
    if pdfs is not None:
        for pdf in pdfs:
            # Read the PDF file and extract text from its pages
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Set up the text splitter for splitting texts into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            # Split the extracted text into chunks for efficient processing
            chunks = text_splitter.split_text(text)

            # Create embeddings and build a knowledge base for the chunks
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Allow the user to input a question about the PDF
        user_question = st.text_input("Ask a question about your PDF")

        # Check if a user question has been entered
        if user_question:
            # Perform similarity search on the knowledge base using the user's question
            docs = knowledge_base.similarity_search(user_question)

            # Set up a question-answering chain
            chain = load_qa_chain(llm, chain_type="stuff")

            # Generate a response to the user's question using the question-answering chain
            response = chain.run(input_documents=docs, question=user_question)

            # Display the generated response
            st.write(response)

if __name__ == '__main__':
    main()
