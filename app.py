import streamlit as st
import os
import time
from rag_pipeline import setup_rag_pipeline, get_response

# Set page config
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# API key handling in session state
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False

# App title and description
st.title("📚 PDF RAG Assistant")
st.markdown("""
Upload up to 4 PDF documents, then ask questions about their content.
The app will find relevant chunks from your documents and generate answers using a cloud-based LLM via API.
""")

# File uploader and API configuration in sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key configuration
    st.subheader("LLM API Setup")
    api_option = st.selectbox(
        "Select LLM Provider", 
        options=["OpenAI", "Anthropic", "Groq"]
    )
    
    if api_option == "OpenAI":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.api_key_configured = True
    elif api_option == "Anthropic":
        api_key = st.text_input("Enter Anthropic API Key", type="password") 
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            st.session_state.api_key_configured = True
    elif api_option == "Groq":
        api_key = st.text_input("Enter Groq API Key", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.session_state.api_key_configured = True
    
    # Embeddings configuration section
    st.subheader("Embeddings API Setup")
    embedding_option = st.selectbox(
        "Select Embeddings Provider", 
        options=["OpenAI", "Cohere", "FastEmbed (Local)", "Ollama (Local)"]
    )
    
    if embedding_option == "OpenAI":
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            openai_emb_key = st.text_input("Enter OpenAI API Key for embeddings", type="password")
            if openai_emb_key:
                os.environ["OPENAI_API_KEY"] = openai_emb_key
                st.success("OpenAI embeddings configured")
        else:
            st.success("Using OpenAI API key for embeddings")
    
    elif embedding_option == "Cohere":
        cohere_key = st.text_input("Enter Cohere API Key", type="password")
        if cohere_key:
            os.environ["COHERE_API_KEY"] = cohere_key
            st.success("Cohere embeddings configured")
            
    elif embedding_option == "FastEmbed (Local)":
        st.info("FastEmbed will be used locally. No API key required.")
        # Check if fastembed is installed
        try:
            import fastembed
            st.success("FastEmbed is installed and ready to use")
        except ImportError:
            st.error("FastEmbed is not installed. Please run: pip install fastembed")
            
    elif embedding_option == "Ollama (Local)":
        st.info("Ollama will be used locally. Make sure Ollama is running.")
        # Check if langchain-ollama is installed
        try:
            import langchain_ollama
            st.success("Langchain-Ollama integration is installed")
        except ImportError:
            st.error("Langchain-Ollama is not installed. Please run: pip install langchain-ollama")
            st.markdown("[Download Ollama](https://ollama.com/download)")
    
    st.markdown("---")
    
    # Document upload section
    st.header("Upload Documents")
    st.markdown("Upload up to 4 PDF files")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="You can upload up to 4 PDF files"
    )

    if uploaded_files:
        if len(uploaded_files) > 4:
            st.error("Please upload a maximum of 4 PDF files.")
            uploaded_files = uploaded_files[:4]
        
        st.success(f"{len(uploaded_files)} files uploaded successfully!")
        file_names = [file.name for file in uploaded_files]
        st.write("Uploaded files:")
        for name in file_names:
            st.write(f"- {name}")
        
        # Button to process files
        if st.button("Process Documents"):
            if not st.session_state.api_key_configured:
                st.error("Please configure an LLM API key first.")
            elif embedding_option == "Cohere" and "COHERE_API_KEY" not in os.environ:
                st.error("Please provide a Cohere API key for embeddings.")
            elif embedding_option == "OpenAI" and "OPENAI_API_KEY" not in os.environ:
                st.error("Please provide an OpenAI API key for embeddings.")
            else:
                with st.spinner("Processing documents... This may take a while."):
                    try:
                        # Remove OpenAI API key if it's causing quota errors
                        if "OPENAI_API_KEY" in os.environ and embedding_option != "OpenAI":
                            del os.environ["OPENAI_API_KEY"]

                        if embedding_option != "Cohere" and "COHERE_API_KEY" in os.environ:
                            del os.environ["COHERE_API_KEY"]

                        # Create RAG chain
                        st.session_state.rag_chain = setup_rag_pipeline(uploaded_files, embedding_option, api_option)
                        st.success("Documents processed! You can now ask questions.")

                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.error("Try selecting a different embeddings provider from the sidebar.")

# Display chat interface
st.header("Chat")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check if API and RAG chain are initialized
    if not st.session_state.api_key_configured:
        with st.chat_message("assistant"):
            st.write("Please configure an LLM API key in the sidebar first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please configure an LLM API key in the sidebar first."})
    elif st.session_state.rag_chain is None:
        with st.chat_message("assistant"):
            st.write("Please upload and process documents first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload and process documents first."})
    else:
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = get_response(prompt, st.session_state.rag_chain)
                    st.write(response)
                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add some info about the app at the bottom
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About this app
This app uses:
- LangChain for the RAG pipeline
- Cloud-based LLMs (OpenAI, Anthropic, or Groq)
- ChromaDB for vector storage
- Multiple embedding options (OpenAI, Cohere, FastEmbed, or Ollama)
""")