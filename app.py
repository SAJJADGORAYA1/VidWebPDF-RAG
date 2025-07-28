# app.py
import streamlit as st
import time
import os
from utils import extract_youtube_id, get_youtube_transcript, extract_text_from_pdf, check_ollama_connection
from rag import process_content, create_rag_chain
from langchain_community.document_loaders import WebBaseLoader

# Streamlit UI Configuration
st.set_page_config(
    page_title=" YouTube & PDF RAG Assistant",
    page_icon="📺",
    layout="centered"
)

# App Header
st.title(" Document Knowledge Assistant")
st.markdown("""
**Local RAG pipeline powered by Ollama on your MacBook M2**  
Extract insights from YouTube videos, web pages, and PDF documents
""")
st.divider()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "content_loaded" not in st.session_state:
    st.session_state.content_loaded = False
if "transcript_error" not in st.session_state:
    st.session_state.transcript_error = False
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "ollama_available" not in st.session_state:
    st.session_state.ollama_available = False

# Sidebar for settings
with st.sidebar:
    st.header(" Configuration")
    
    # Model selection
    model_options = ["phi3:mini", "mistral:7b-instruct"]
    selected_model = st.selectbox(
        "Ollama Model", 
        model_options,
        index=0,
        help="phi3:mini (fastest), mistral:7b (better quality)"
    )
    
    # Input type
    input_type = st.radio("Content Source", ["YouTube Video", "Web Page", "PDF Document"])
    
    if input_type == "YouTube Video":
        video_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=Gfr50f6ZBvo")
    elif input_type == "Web Page":
        web_url = st.text_input("Web Page URL", "https://en.wikipedia.org/wiki/Large_language_model")
    else:
        uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_pdf:
            st.session_state.pdf_uploaded = True
    
    # Performance settings
    with st.expander("Performance Settings"):
        k_value = st.slider("Retrieval Chunks", 2, 6, 3)
        chunk_size = st.slider("Chunk Size", 300, 1000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 50, 300, 100)
        temperature = st.slider("Creativity", 0.0, 1.0, 0.3)
        max_tokens = st.slider("Max Response Tokens", 100, 512, 256)
    
    st.divider()
    st.subheader("System Status")
    status_indicator = st.empty()
    
    # Check Ollama connection
    if check_ollama_connection():
        status_indicator.success(" Ollama server is running")
        st.session_state.ollama_available = True
    else:
        status_indicator.error(" Ollama server not detected")
        st.info("To run Ollama server:")
        st.code("ollama serve", language="bash")
        st.session_state.ollama_available = False
    
    st.divider()
    st.caption(f"Running locally on {os.uname().machine}")

# Load content based on input type
def load_content(input_type, url=None, uploaded_file=None):
    try:
        if input_type == "YouTube Video":
            video_id = extract_youtube_id(url)
            if not video_id:
                return None, "Invalid YouTube URL. Please use a valid URL with 11-character ID"
            
            transcript, error = get_youtube_transcript(video_id)
            if error:
                return None, error
            elif transcript:
                return " ".join(chunk["text"] for chunk in transcript), None
            else:
                return None, "Failed to retrieve transcript"
        elif input_type == "Web Page":
            loader = WebBaseLoader(url)
            return loader.load()[0].page_content, None
        else:  # PDF Document
            if not uploaded_file:
                return None, "No PDF file uploaded"
            return extract_text_from_pdf(uploaded_file)
    except Exception as e:
        return None, str(e)

# Main application logic
def main():
    # Check Ollama status before proceeding
    if not st.session_state.ollama_available:
        st.error("""
        ## Ollama Server Not Detected
        Please start the Ollama server before using this application:
        
        ```bash
        ollama serve
        ```
        
        If you haven't installed Ollama yet, download it from [ollama.ai](https://ollama.ai/)
        """)
        return
        
    # Show initialization button if not initialized
    if not st.session_state.initialized:
        st.subheader("Step 1: Initialize System")
        st.write("Configure settings in the sidebar and click the button below to start.")
        
        # URL examples and caption info
        with st.expander(" Important Notes"):
            st.markdown("""
            ### For YouTube Videos:
            1. **Must have captions enabled** - This app uses YouTube's captions
            2. **Supported URL formats**:
               - Full URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
               - Short URL: `https://youtu.be/dQw4w9WgXcQ`
               - Video ID: `dQw4w9WgXcQ`
            3. **Test with known good videos**:
               - [AI Explained](https://www.youtube.com/watch?v=Gfr50f6ZBvo)
               - [Tech Overview](https://www.youtube.com/watch?v=aircAruvnKk)
            
            ### For PDF Documents:
            - Works with text-based PDFs (not scanned/image-based)
            - File size limit: 10MB
            - Document length limit: 50 pages
            
            ### Ollama Requirements:
            - Must have Ollama running locally (`ollama serve`)
            - Required models: `phi3:mini` or `mistral:7b-instruct`
            """)
        
        if st.button(" Initialize RAG System", type="primary", use_container_width=True):
            # Validate input
            if input_type == "YouTube Video":
                if not video_url:
                    st.error("Please enter a YouTube URL")
                    return
                content, error = load_content("YouTube Video", url=video_url)
            elif input_type == "Web Page":
                if not web_url:
                    st.error("Please enter a Web Page URL")
                    return
                content, error = load_content("Web Page", url=web_url)
            else:  # PDF Document
                if not uploaded_pdf:
                    st.error("Please upload a PDF file")
                    return
                content, error = load_content("PDF Document", uploaded_file=uploaded_pdf)
            
            if error:
                st.error(f"Error: {error}")
                st.session_state.transcript_error = True
                
                # Special handling for caption issues
                if "No English captions" in error or "rate limit" in error:
                    st.info("Try these solutions:")
                    st.markdown("""
                    -  Wait 1-2 minutes and try again
                    -  Use a different video URL
                    -  Test with these working videos:
                      - AI Explained: `https://www.youtube.com/watch?v=Gfr50f6ZBvo`
                      - Neural Networks: `https://www.youtube.com/watch?v=aircAruvnKk`
                    """)
                return
            
            if content:
                # Process content and build vector store
                with st.spinner(" Building knowledge base (this may take a few minutes)..."):
                    try:
                        st.session_state.vector_store = process_content(
                            content, 
                            chunk_size, 
                            chunk_overlap,
                            selected_model
                        )
                        
                        if not st.session_state.vector_store:
                            st.error("Failed to create vector store")
                            return
                            
                        st.session_state.content_loaded = True
                        st.session_state.transcript_error = False
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        st.session_state.content_loaded = False
                        return
                
                st.session_state.initialized = True
                st.success("System initialized successfully!")
                st.experimental_rerun()  # Rerun to refresh the UI
    
    elif st.session_state.initialized and st.session_state.content_loaded:
        # System is initialized, show the chat interface
        st.subheader("Step 2: Chat with the Content")
        
        # Initialize RAG chain with error handling
        rag_chain = create_rag_chain(
            st.session_state.vector_store, 
            k_value,
            selected_model,
            temperature,
            max_tokens
        )
        
        if not rag_chain:
            st.error("Failed to initialize RAG chain. Check Ollama connection.")
            return
            
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the document/video"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner(" Generating response..."):
                    start_time = time.time()
                    try:
                        response = rag_chain.invoke({"query": prompt})
                        
                        if "result" not in response:
                            raise ValueError("Invalid response from RAG chain")
                            
                        elapsed = time.time() - start_time
                        st.markdown(response["result"])
                        
                        # Show source context
                        with st.expander(" See source context"):
                            if response.get("source_documents"):
                                for i, doc in enumerate(response["source_documents"]):
                                    st.caption(f"Source chunk {i+1}:")
                                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            else:
                                st.info("No source documents retrieved")
                        
                        st.caption(f"Generated in {elapsed:.2f}s | Model: {selected_model}")
                        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        if "HTTPConnectionPool" in str(e) or "Connection refused" in str(e):
                            st.error("Ollama connection lost. Please restart the server with `ollama serve`")
        
        # Add reset button
        if st.button(" Reset System", type="secondary"):
            st.session_state.clear()
            st.experimental_rerun()
    
    elif st.session_state.initialized and not st.session_state.content_loaded:
        st.warning("Content failed to load. Please try another source.")
        if st.session_state.transcript_error:
            st.info("YouTube rate limits are temporary. Wait 1-2 minutes and try again.")
        if st.button(" Back to Setup"):
            st.session_state.initialized = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()