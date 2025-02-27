# AI-Powered Q&A System with RAG and FAISS

## 📌 Overview
This project implements an AI-powered **Question & Answer (Q&A) System** using **Large Language Models (LLMs), embeddings, and Retrieval-Augmented Generation (RAG)**. It ingests documents, generates embeddings, stores them in **FAISS**, and retrieves relevant context to answer user queries.

## 🚀 Features
- **Document Ingestion:** Supports `.txt` and `.pdf` files.
- **Vector Search:** Uses **FAISS** for fast document retrieval.
- **Embeddings:** Utilizes **OpenAI's text-embedding-ada-002**.
- **RAG-Based Answer Generation:** Retrieves relevant document context before generating responses.
- **Streamlit UI:** User-friendly web interface for asking questions.
- **Scalable & Efficient:** Optimized retrieval and response generation pipeline.

## 🏗️ Project Structure
```
📂 NOEV
├── 📂 docs                # Document storage              
├── app.py                 # Streamlit UI
├── main.py                # Main entry point for backend processing
├── requirements.txt       # Python dependencies
├── packages.txt           # Additional packages list
├── runtime.txt            # Runtime configuration
├── README.md              # Project documentation
```

## 📥 Installation
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/AnassTalha/QA_system.git
cd QA_system
```
### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔄 Usage
### 1️⃣ Run the Streamlit UI
```bash
streamlit run app.py
```
This will launch the **interactive Q&A interface** in your web browser.

## 🔍 How It Works
1. **Document Processing:** Reads `.txt` and `.pdf` files, extracts text, and chunks content.
2. **Embedding Generation:** Uses **OpenAI embeddings** to vectorize document chunks.
3. **Vector Storage:** Stores embeddings in **FAISS** for fast retrieval.
4. **Query Processing:** User inputs a question, and the system retrieves the top relevant chunks.
5. **Answer Generation:** The retrieved chunks are fed to an **LLM**, which generates a response.

## 🎯 Example Query
```
User: "What are the key findings of the report?"
AI Answer: "The report highlights that ... [context retrieved from documents]."
```

## 📊 Performance & Optimization
- **FAISS Indexing:** Ensures fast and efficient vector retrieval.
- **Chunking Strategy:** Uses overlapping text chunks to improve retrieval accuracy.
- **Hybrid Search (Optional):** Can integrate **BM25 for keyword-based retrieval**.

## 🚀 Deployment
### **Option 1: Deploy on Hugging Face Spaces**
1. Create a new **Hugging Face Space**.
2. Select **Streamlit** as the environment.
3. Upload your repository.
4. Add `requirements.txt` and ensure dependencies are installed.
5. Deploy and get a public link!

### **Option 2: Deploy via FastAPI (Production API)**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
This will launch a FastAPI backend for programmatic access.

## 🛠️ Future Improvements
- ✅ Add support for **ChromaDB / Pinecone** for better scalability.
- ✅ Implement **Hybrid Retrieval (BM25 + FAISS)**.
- ✅ Improve UI with **file-specific search options**.
- ✅ Secure API keys using **environment variables**.

## 💡 Credits
- Built with **LangChain, OpenAI API, FAISS, and Streamlit**.
- Inspired by **Retrieval-Augmented Generation (RAG) architectures**.

## 📜 License
This project is open-source under the **MIT License**.

---
🎯 **Ready to build the best AI Q&A system? Fork this repo and enhance it further!** 🚀


