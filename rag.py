# rag.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
import requests

def process_content(content, chunk_size, chunk_overlap, model_name):
    """Process content and create vector store"""
    # Memory-efficient text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len
    )
    chunks = text_splitter.split_text(content)
    
    # Initialize embeddings
    try:
        embeddings = OllamaEmbeddings(
            model=model_name,
            show_progress=True
        )
        
        # Test the connection with a small embedding
        test_embedding = embeddings.embed_query("test connection")
        if not test_embedding:
            raise ConnectionError("Ollama returned empty embedding")
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
    
    # Create vector store
    try:
        return FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")

def init_llm(model, temperature, max_tokens):
    """Initialize Ollama LLM with connection check"""
    try:
        # Test connection by listing models
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise ConnectionError(f"Ollama API returned status {response.status_code}")
            
        return Ollama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            system="Answer concisely using ONLY the context. If unsure, say 'I don't know'.",
        )
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Ollama LLM: {str(e)}")

def create_rag_chain(vector_store, k_value, model, temperature, max_tokens):
    """Create RAG chain with error handling"""
    llm = init_llm(model, temperature, max_tokens)
    if not llm:
        return None
        
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": min(k_value, 3)}  # Conservative for 8GB RAM
    )
    
    # Memory-efficient prompt
    prompt_template = """
    Context: {context}
    Question: {question}
    Answer: 
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )