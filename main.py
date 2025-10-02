import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load config
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error("Groq API Key not found. Please set it in your .env file.")
    st.stop()

st.set_page_config(layout="wide", page_title="Groq RAG Streamlit")

# Initialize session state for vectorstore
if 'vectorstore' not in st.session_state:
    with st.spinner("Loading documents and building embeddings..."):
        st.session_state.embeddings = HuggingFaceInstructEmbeddings()
        st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Documents loaded and vectorstore ready!")

# Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-20b")

# Prompt template
# Prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Create chains
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# UI
st.title("Groq-Powered RAG Chat")
st.markdown("Enter your query below and get answers based on the loaded documents:")

prompt_input = st.text_input("Ask a question:")

if prompt_input:
    start_time = time.time()
    try:
        response = retrieval_chain.invoke({"input": prompt_input})
        elapsed_time = time.time() - start_time

        # Display answer
        st.subheader("Answer:")
        st.write(response.get('answer', "No answer found."))

        st.info(f"Response time: {elapsed_time:.2f} seconds")

        # Show relevant documents in expander
        if 'context' in response:
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response['context']):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")
        else:
            st.warning("No context chunks returned.")

    except Exception as e:
        st.error(f"Error during retrieval: {e}")
