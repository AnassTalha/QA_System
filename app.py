import os
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("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")

# Initialize Hugging Face embeddings model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Initialize Hugging Face Q&A pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

# Load documents
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip non-text files like requirements.txt
        if filename.endswith(".txt") and filename != "requirements.txt":
            loader = TextLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue  # Skip invalid files

        documents.extend(loader.load())

    return documents


# Create embeddings index
def create_embedding_index(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    text_data = [doc.page_content for doc in texts]
    vector_db = FAISS.from_texts(text_data, embeddings)
    return vector_db

# Retrieve relevant documents based on user query
def retrieve_documents(query, index, top_k=3):
    return [doc.page_content for doc in index.similarity_search(query, k=top_k)]

# Generate AI response
def generate_response(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    result = qa_pipeline({'question': query, 'context': context})
    return result['answer']

# Streamlit UI
st.title("AI-Powered Q&A System")
st.write("Ask a question related to the uploaded documents!")

# Upload documents
uploaded_files = st.file_uploader("Choose some files", accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        # Save file to disk and load
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        docs.extend(load_documents("."))  # Load documents

    # Create FAISS index for documents
    index = create_embedding_index(docs)

    # User query input
    query = st.text_input("Enter your question:")
    if query:
        retrieved_docs = retrieve_documents(query, index)
        answer = generate_response(query, retrieved_docs)
        st.write("✅ AI Answer: ", answer)



