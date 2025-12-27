import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader

# --- IMPORTS ---

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq 
from langchain.chains import ConversationalRetrievalChain

# Load API Keys
load_dotenv()

def main():
    st.set_page_config(page_title="Chat PDF")
    st.title("AI PDF Chatbot 🤖")

    # --- Session State (Memory) ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # --- Upload Section ---
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                # Read PDF
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                # Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                
                # Embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                # Create Vector Store
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                st.success("PDF Processed Successfully!")

    # --- Chat Section ---
    question = st.text_input("Ask a question about the PDF")

    if question:
        if st.session_state.vector_store is None:
            st.error("Please upload and process a PDF first!")
        else:
            # Setup LLM
            llm = ChatGroq(
                temperature=0, 
                groq_api_key=os.getenv("API_KEY"), 
               model_name="llama-3.3-70b-versatile"
            )
            
            # Create Chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                retriever=st.session_state.vector_store.as_retriever()
            )
            
            # Run Question
            response = qa_chain.invoke({
                "question": question, 
                "chat_history": st.session_state.chat_history
            })
            
            # Display
            st.session_state.chat_history.append((question, response["answer"]))
            st.write("**Answer:**", response["answer"])

if __name__ == "__main__":
    main()