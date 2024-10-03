import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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
Given the following context from the documents, and previous interactions, answer the user's new question. Provide clear and accurate responses.
Context from Documents:
<context>
{context}
<context>
Previous Interactions:
{conversation_history}
New Question: {input}
""")

# Load embeddings
def load_embeddings():
    try:
        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        final_documents = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(final_documents, embeddings)
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
