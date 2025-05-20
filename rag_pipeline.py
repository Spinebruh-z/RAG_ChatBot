import os
import tempfile
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, CohereEmbeddings, FastEmbedEmbeddings

def get_embeddings(embedding_provider):
    """
    Get an appropriate embeddings model based on available APIs
    """
    # Try different embedding options in order of preference
    if os.getenv("OPENAI_API_KEY") and embedding_provider == "OpenAI":
        print("Using OpenAI embeddings")
        try:
            # Option 1: OpenAI embeddings (most reliable)
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            print(f"Failed to initialize OpenAI embeddings: {e}")

    if os.getenv("COHERE_API_KEY") and embedding_provider == "Cohere":
        print("Using Cohere embeddings")
        try:
            # Option 2: Cohere embeddings
            return CohereEmbeddings(model="embed-english-v3.0")
        except Exception as e:
            print(f"Failed to initialize Cohere embeddings: {e}")
    
    # Option 3: Lightweight local FastEmbed embeddings (no API key required)
    try:
        print("Using FastEmbed embeddings")
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        print(f"Failed to initialize FastEmbed embeddings: {e}")
    
    # Last resort: Try Ollama if installed locally
    try:
        print("Using Ollama embeddings")
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        print(f"Failed to initialize Ollama embeddings: {e}")
        
    # If all else fails, raise an error with guidance
    raise ValueError(
        "Could not initialize any embeddings model. Please set one of the following environment variables: "
        "OPENAI_API_KEY, COHERE_API_KEY, or install FastEmbed/Ollama locally."
    )

def load_llm(llm_provider):
    """
    Load a cloud-based LLM using API
    """
    # Check which API token is available
    if os.getenv("OPENAI_API_KEY") and llm_provider == "OpenAI":
        # OpenAI option
        return ChatOpenAI(
            model="gpt-3.5-turbo",  # Affordable option for most use cases
            temperature=0.7
        )
    elif os.getenv("ANTHROPIC_API_KEY") and llm_provider == "Anthropic":
        # Anthropic option
        return ChatAnthropic(
            model_name="claude-3-haiku-20240307",  # Lightweight Claude model
            temperature=0.7,
            max_tokens_to_sample=512,
            timeout=60,
            stop=None
        )
    elif os.getenv("GROQ_API_KEY") and llm_provider == "Groq":
        # Groq option (very fast inference)
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=512,
        )
    else:
        raise ValueError(
            "No API keys found. Please set one of the following environment variables: "
            "HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY"
        )

def setup_rag_pipeline(pdf_files, embedding_provider, llm_provider):
    """
    Set up the RAG pipeline with the provided PDF files
    """
    # Load and chunk the documents
    documents = []
    temp_files = []
    
    try:
        for file in pdf_files:
            # Create a temporary file with a unique name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp_path = temp.name
                temp_files.append(temp_path)
                # Write PDF content to the temp file
                temp.write(file.read())
                # Close the file explicitly before loading
                temp.flush()
                temp.close()
                
            # Now load the PDF with PyMuPDFLoader
            loader = PyMuPDFLoader(temp_path)
            documents.extend(loader.load())

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Create embeddings and vector database
        try:
            # Get embeddings based on available APIs
            embeddings = get_embeddings(embedding_provider)
            
            # Create vector database
            vectordb = Chroma.from_documents(chunks, embedding=embeddings)
            
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Load the LLM
            llm = load_llm(llm_provider)
            
        except Exception as e:
            raise RuntimeError(f"Failed to set up embeddings, vector DB, or LLM: {e}")

        # Prompt template for RAG
        prompt_template = """You are a helpful assistant. Use the context below to answer the user's question.
If the answer isn't in the context, say "I couldn't find that in the documents."

Context:
{context}

Question:
{question}

Answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # Create the RAG chain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return rag_chain
        
    finally:
        # Clean up temp files in the finally block to ensure it happens
        # even if there's an exception
        import time

        # Small delay to ensure files are fully released
        time.sleep(0.1)
        
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                # Just log the error but don't raise it to avoid breaking the app
                print(f"Warning: Could not delete temporary file {temp_path}: {e}")

def get_response(query, rag_chain):
    """
    Get a response from the RAG chain for the given query
    """
    return rag_chain.run({"question": query})