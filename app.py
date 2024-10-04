import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pickle

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize Streamlit app
st.title("Ask In Case!")

# Create LLM model instance
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template, making it more conversational
prompt = ChatPromptTemplate.from_template("""
You're an AI assistant here to help answer questions based on the provided documents and previous conversations. Be friendly and conversational in your responses. And you you can refer to as Database instead of saying documents in chats. You can you online support incase needed to answer a question. Your name is Ask in Case.

Documents:
{context}

Previous Conversations:
{conversation_history}

New Question: {input}
""")

# Function to save and load FAISS index to avoid recomputation
def save_faiss_index(faiss_store, filename="faiss_index.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(faiss_store, f)

def load_faiss_index(filename="faiss_index.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# Function to load documents and create vector database, optimized for faster embeddings
@st.cache_resource(show_spinner=False)
def load_embeddings():
    try:
        with st.spinner("Loading and processing documents..."):
            # Check if the FAISS index already exists
            vector_store = load_faiss_index()
            if vector_store:
                st.write("Loaded precomputed embeddings from cache.")
                return vector_store

            # Load all PDFs in the directory
            loader = PyPDFDirectoryLoader("./us_census")
            docs = loader.load()
            st.write(f"Loaded {len(docs)} documents.")

            # Document chunking with adjusted chunk size and overlap
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)

            # Use Hugging Face embeddings (smaller, faster model)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Parallel processing to speed up embedding computation
            with ThreadPoolExecutor() as executor:
                chunks = [final_documents[i:i + 50] for i in range(0, len(final_documents), 50)]  # Chunk size of 50 docs
                futures = [executor.submit(FAISS.from_documents, chunk, embeddings) for chunk in chunks]
                stores = [future.result() for future in futures]

            # Merge the FAISS stores into one
            vector_store = stores[0]
            for store in stores[1:]:
                vector_store.merge_from(store)

            # Cache the FAISS index to avoid recomputation
            save_faiss_index(vector_store)
            st.write("Document embeddings created and saved successfully.")
            return vector_store
    except Exception as e:
        st.error(f"Error during embedding: {e}")
        return None

# Load or retrieve embeddings
if "vectors" not in st.session_state:
    st.session_state.vectors = load_embeddings()

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Text input for the question
prompt1 = st.text_input("Ask your question:")

# Add question to conversation history
if st.button("Submit"):
    if prompt1.lower() in ["hello", "hi", "hey"]:
        st.session_state.conversation_history.append(f"User: {prompt1}")
        st.session_state.conversation_history.append("Model: Hi there! How can I assist you today?")
        st.write("Hi there! How can I assist you today?")
    else:
        st.session_state.conversation_history.append(f"User: {prompt1}")

        try:
            # Create a retrieval chain for querying
            with st.spinner("Processing your question..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Retrieve top 5 similar docs
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Prepare context for the new query
                conversation_history = "\n".join(st.session_state.conversation_history)
                response = retrieval_chain.invoke({
                    'input': prompt1,
                    'conversation_history': conversation_history,
                    'context': retriever  # Ensure context from documents is passed
                })

                # Store the model's response
                st.session_state.conversation_history.append(f"Model: {response['answer']}")

                # Display response
                st.write(response['answer'])

                # Display document similarity search results in an expander
                with st.expander("Relevant Document Snippets"):
                    for index, doc in enumerate(response["context"][:5]):  # Show top 5 relevant documents
                        similarity_score = doc.metadata.get('similarity_score', 'N/A')
                        st.write(f"Source: {doc.metadata.get('source', 'Unknown')} (Similarity score: {similarity_score})")

                        snippet = doc.page_content[:min(500, len(doc.page_content))]  # Show excerpt of 500 characters
                        st.text_area(f"Document Excerpt {index}", snippet, height=200, key=f"text_area_{index}")  # Unique key

                        if len(doc.page_content) > 500:
                            if st.button(f"Show full content for document {doc.metadata.get('source', 'Unknown')}", key=f"button_{index}"):
                                st.write(doc.page_content)

                        st.write("--------------------------------")

        except Exception as e:
            st.error(f"Error during document retrieval: {e}")

# Display conversation history
st.header("Conversation History")
for item in st.session_state.conversation_history:
    st.write(item)
