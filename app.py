import streamlit as st
import os
import time
import math
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

# Initialize Streamlit app
st.title("Ask In Cases!")

# Create LLM model instance
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template, making sure to include "context"
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

# Function to load documents and embeddings, cached for efficiency
@st.cache_resource(show_spinner=False)
def load_embeddings():
    try:
        with st.spinner("Loading and processing documents..."):
            loader = PyPDFDirectoryLoader("./us_census")  # Load all PDFs in the directory
            docs = loader.load()
            st.write(f"Loaded {len(docs)} documents.")

            # Document chunking with adjusted chunk size and overlap
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            final_documents = text_splitter.split_documents(docs)
            total_requests = len(final_documents)

            # Number of requests you can send per minute
            requests_per_minute = 150
            batches = math.ceil(total_requests / requests_per_minute)  # Number of batches needed

            # Calculate estimated time in minutes
            estimated_time = batches  # 1 minute per batch
            st.write(f"Total requests: {total_requests}")
            st.write(f"Estimated time to process all requests: {estimated_time} minutes")

            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = None

            # Process documents in batches
            for i in range(batches):
                # Determine the range of documents to process in this batch
                start_index = i * requests_per_minute
                end_index = min(start_index + requests_per_minute, total_requests)
                batch_docs = final_documents[start_index:end_index]

                try:
                    # Process batch and create embeddings
                    vector_store = FAISS.from_documents(batch_docs, embeddings)
                    st.write(f"Processed batch {i+1}/{batches} (documents {start_index + 1} to {end_index})")

                    if i < batches - 1:  # Sleep only between batches, not after the last one
                        time.sleep(60)  # Wait for 1 minute to stay within the limit

                except Exception as e:
                    st.error(f"Error during embedding in batch {i+1}: {e}")

            st.write("Document embeddings created successfully.")
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
    st.session_state.conversation_history.append(f"User: {prompt1}")

    try:
        # Create a retrieval chain for querying
        with st.spinner("Processing your question..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
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
                    st.text_area(f"Document Excerpt {index+1}", snippet, height=200, key=f"snippet_{index}")  # Unique key

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
