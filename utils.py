# utils.py
import re
import random
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from PyPDF2 import PdfReader
from io import BytesIO

def extract_youtube_id(url):
    """Extract YouTube ID from various URL formats"""
    # Standard watch URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    if "youtube.com/watch" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None
    
    # Shortened URL: https://youtu.be/dQw4w9WgXcQ
    if "youtu.be" in url:
        match = re.search(r"youtu.be/([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None
    
    # Embed URL: https://www.youtube.com/embed/dQw4w9WgXcQ
    if "youtube.com/embed" in url:
        match = re.search(r"embed/([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None
    
    # Return as-is if it's already an ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url
    
    return None

def get_youtube_transcript(video_id, max_retries=3):
    """Retrieve YouTube transcript with retries and backoff"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    
    for attempt in range(max_retries):
        try:
            # Create session with custom headers
            session = requests.Session()
            session.headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Get transcript using the custom session
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en'],
                session=session
            )
            
            if not transcript:
                return None, " Transcript is empty or not found."
            return transcript, None
        except (TranscriptsDisabled, NoTranscriptFound):
            return None, " No transcript available for this video."
        except Exception as e:
            if "no element found" in str(e).lower():
                return None, " YouTube returned empty response (likely blocked or unavailable)."
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f" Failed to retrieve captions: {str(e)}"

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    try:
        # Create a PDF reader object
        pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
        
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            return None, "PDF appears to be image-based (no text extracted)"
        
        return text, None
    except Exception as e:
        return None, f"PDF processing error: {str(e)}"

def check_ollama_connection():
    """Check if Ollama server is running"""
    try:
        import socket
        # Try to connect to Ollama server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(("localhost", 11434))
        return True
    except:
        return False