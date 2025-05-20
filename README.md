# 📚 PDF RAG Assistant

A powerful **Retrieval-Augmented Generation (RAG)** app built with **LangChain**, **Streamlit**, and **FAISS**, allowing users to upload PDF documents and query them using top LLMs like **OpenAI**, **Anthropic**, and **Groq** — with support for various embedding providers.

## 🔍 Features

- ✅ Upload up to 4 PDFs at once
- ✅ Query document contents using conversational LLMs
- ✅ Uses semantic search over document chunks
- ✅ Supports memory via conversational history
- ✅ Choose from multiple LLMs:
  - OpenAI (GPT-3.5)
  - Anthropic (Claude Haiku)
  - Groq (LLaMA 3.1)
- ✅ Multiple embedding options:
  - OpenAI
  - Cohere
  - FastEmbed (local)
  - Ollama (local)
- ✅ All API keys are provided securely through the UI
- ✅ Fully deployable on **Streamlit Cloud**

---

## 🚀 Live Demo

> 🔗 https://rag-langchain-chat-bot.streamlit.app/

---

## 🖥️ How It Works

1. **User uploads PDFs** via sidebar.
2. Documents are chunked and embedded using the selected embedding model.
3. FAISS builds an in-memory vector index.
4. On each user query:
   - Top 5 relevant chunks are retrieved.
   - Combined with the query and sent to the LLM.
   - Response is returned and appended to the chat history.

---

## 🛠️ Tech Stack

- `Streamlit` – UI and user input
- `LangChain` – RAG pipeline + memory
- `FAISS` – Fast in-memory vector search
- `PyMuPDF` – PDF parsing
- `sentence-transformers`, `fastembed`, `langchain-ollama` – Embeddings
- `OpenAI`, `Groq`, `Anthropic` – LLM APIs

---

Created By - Sarthak S. Satam
