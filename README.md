#  YouTube & PDF RAG Assistant



A privacy-focused, locally running **Retrieval-Augmented Generation (RAG)** system powered by **Ollama** and optimized for **Apple Silicon (M1/M2/M3)**. This assistant allows you to **extract knowledge** from YouTube videos (via transcripts), PDF documents, and web pages — and ask questions in natural language.

---

##  Overview

This app combines **modern LLM techniques** with **local vector search** to deliver fast, private, and explainable responses using real document context.

###  Key Features
-  **YouTube Transcript Retrieval**  
-  **PDF Text Extraction**  
-  **Web Page Content Scraping**  
-  **LangChain-based Vector Search (FAISS)**  
-  **LLM-Powered Q&A with RAG Chain**  
-  **Local Deployment with Ollama**  
-  **Data Privacy – Everything runs on your machine**

---

##  Core Concepts Explained

###  Retrieval-Augmented Generation (RAG)
RAG is a framework where **external knowledge is retrieved first** and then passed to a language model for answer generation. This helps reduce hallucination and ensures answers are **grounded in source documents**.

In this app:
1. **Content** (YouTube transcript / PDF / web page) is chunked and embedded.
2. A **vector store** (FAISS) is created.
3. Your query retrieves **top-k relevant chunks** from the vector store.
4. These chunks are passed to an LLM (like `phi3:mini` or `mistral`) for generating answers.

---

###  Vector Embeddings & Semantic Search
- The **`OllamaEmbeddings`** class generates high-dimensional vector representations of text using your chosen local model.
- Similarity search uses **FAISS**, a fast approximate nearest neighbor library.
- **Query vector** is compared against **document vectors** to find the most relevant chunks.

###  Chunking & Text Splitting
Text is split using **LangChain’s RecursiveCharacterTextSplitter**, preserving meaning and ensuring context is retained across chunk boundaries.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

This helps the retriever deliver coherent document segments to the LLM.

---

###  LLM Integration with Ollama
- Uses **Ollama's local server** (`ollama serve`) to run LLMs like `phi3:mini` or `mistral:7b-instruct` directly on Apple Silicon.
- No cloud APIs are involved — ideal for **offline and secure setups**.
- Streamlit checks connectivity to `localhost:11434` to ensure Ollama is running.

```python
Ollama(
  model=model_name,
  temperature=0.3,
  num_predict=256,
)
```

---

###  Prompt Engineering & Contextual QA
Prompt format is minimal and task-specific:

```text
Context: {retrieved documents}
Question: {your query}
Answer:
```

This design ensures the model focuses **only on grounded context** and avoids hallucinations.

---

##  Installation Guide

###  Requirements
- macOS with M1/M2/M3 chip
- Python 3.10+
- [Ollama](https://ollama.ai) installed locally
- At least 8GB RAM (recommended)

### Setup Instructions

```bash
# Clone the repo
git clone https://github.com/SAJJADGORAYA1/VidWebPDF-RAG

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

###  Install Ollama Models
```bash
# Start the Ollama server
ollama serve

# Pull the models
ollama pull phi3:mini
ollama pull mistral:7b-instruct
```

---

##  Running the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

##  Performance Tuning

- **Chunk Size**: Number of characters per text segment (500 is ideal)
- **Chunk Overlap**: Overlap between chunks to preserve context
- **Retrieval Chunks (k)**: Number of chunks retrieved per question (k=3 is memory safe)
- **Temperature**: Controls randomness in answers
- **Max Tokens**: Controls length of LLM-generated responses

---

##  Security & Privacy

This assistant is **fully local**:
- No data leaves your device
- No third-party API calls
- Works offline (after model download)

---


##  Tech Stack

| Component              | Tech Used                  |
|------------------------|----------------------------|
| UI                     | Streamlit                  |
| LLM                    | Ollama (phi3, mistral)     |
| Vector Store           | FAISS                      |
| Embeddings             | OllamaEmbeddings           |
| Chunking & Retrieval   | LangChain Splitters        |
| PDF Reading            | PyPDF2                     |
| YouTube Captions       | youtube-transcript-api     |
| Web Content Loader     | LangChain WebBaseLoader    |

---

##  Further Reading

- [Ollama Documentation](https://ollama.ai)
- [LangChain RAG Docs](https://docs.langchain.com/docs/use-cases/question-answering/)
- [FAISS Paper (Facebook AI)](https://research.facebook.com/publications/faiss-a-library-for-efficient-similarity-search/)

---

## Future Improvements
- OCR for Scanned PDFs using Tesseract or EasyOCR.
- Web Search Integration via Google/Bing/DDG APIs.
- API Key Support for OpenAI, Claude, Gemini, Mistral, etc.
- Connect custom local/remote open-source LLMs.
- Custom Prompt Templates and chat personas.
- Upload Personal Knowledge Base: Notes, docs, and HTML files.
- Project Folders & Tags for organizing documents.
- Chat History & Export in Markdown or JSON.
- Token Usage Tracking for API-based models.
- Combine video, PDF, and web in one query.
- Offline/Online Mode Switching between local Ollama and cloud APIs.
- Scheduled Auto-Refresh of web content and vector stores.
- Voice Input Support for hands-free querying.

---