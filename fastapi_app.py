import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')

# Create FastAPI instance
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM model instance
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
You're an AI assistant here to help answer questions based on the provided documents and previous conversations. Be friendly and conversational in your responses. And you can refer to as Database instead of saying documents in chats. Your name is Ask in Case.

Context from Documents:
{context}

Previous Conversations:
{conversation_history}

New Question: {input}
""")

# Function to load FAISS index to avoid recomputation
def load_faiss_index(filename="faiss_index.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# Load embeddings
def load_embeddings():
    try:
        # Check if the FAISS index already exists
        vector_store = load_faiss_index()
        if vector_store:
            return vector_store

        # Load all PDFs in the directory if FAISS index doesn't exist
        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        final_documents = text_splitter.split_documents(docs)

        # Use Hugging Face embeddings (smaller, faster model)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(final_documents, embeddings)

        # Save the FAISS index to avoid recomputation in the future
        with open("faiss_index.pkl", "wb") as f:
            pickle.dump(vector_store, f)

        return vector_store
    except Exception as e:
        return None

# Store embeddings for document retrieval
vector_store = load_embeddings()

# API route to handle chat requests
@app.post("/chat")
async def chat_with_bot(request: dict):
    user_input = request.get("input")

    try:
        # Create a retrieval chain for querying
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Prepare context for the new query
        response = retrieval_chain.invoke({
            'input': user_input,
            'conversation_history': "",  # If you plan to pass conversation history
            'context': retriever
        })

        return {"answer": response['answer']}
    
    except Exception as e:
        return {"error": str(e)}
