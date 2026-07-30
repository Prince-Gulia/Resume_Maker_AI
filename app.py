import os
import time
import warnings
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract as pyt
from dotenv import load_dotenv

# LangChain & Tavily Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient

# Load environment variables from local .env file
load_dotenv()
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Resume Generator & Job Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-mode aesthetic and crisp UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4A90E2;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #2B6CB0;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #3182CE;
        color: white;
    }
    .card-box {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🔑 API Settings")
st.sidebar.info("Configure your API credentials below. Defaults are loaded automatically if provided in your environment.")

# Retrieve default keys safely from Streamlit secrets or environment variables
DEFAULT_GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
DEFAULT_TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY", ""))

gemini_api_key = st.sidebar.text_input("Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password", placeholder="AIzaSy...")
tavily_api_key = st.sidebar.text_input("Tavily API Key", value=DEFAULT_TAVILY_KEY, type="password", placeholder="tvly-...")
tesseract_path = st.sidebar.text_input("Tesseract OCR Path (Optional)", value="", placeholder="e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

if tesseract_path.strip():
    pyt.pytesseract.tesseract_cmd = tesseract_path.strip()

def extract_text_from_llm_response(response) -> str:
    """Safely extracts text string from various LangChain message response formats."""
    if isinstance(response, str):
        return response
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
            return "\n".join(parts)
    if isinstance(response, dict):
        if 'output' in response:
            return str(response['output'])
        if 'messages' in response and len(response['messages']) > 0:
            return extract_text_from_llm_response(response['messages'][-1])
    return str(response)

def clean_html_code(raw_code: str) -> str:
    """Strips markdown block markers from generated HTML."""
    code = raw_code.strip()
    if code.startswith("```html"):
        code = code[7:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def get_llm_model(api_key: str):
    """Initializes Google Gemini model safely."""
    if not api_key or not api_key.strip():
        st.error("⚠️ Missing Gemini API Key. Please enter a valid key from Google AI Studio (https://aistudio.google.com/).")
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=api_key.strip(),
            temperature=0.2
        )
    except Exception as e:
        st.error(f"Error initializing Gemini Model: {str(e)}")
        return None

def search_tavily_jobs(query: str, api_key: str):
    """Executes search via Tavily Client."""
    if not api_key:
        return "Tavily API key missing."
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="advanced", max_results=10)
        return response
    except Exception as e:
        return f"Error executing Tavily search: {str(e)}"

def generate_resume_html(llm, user_query: str) -> str:
    """Orchestrates system prompt generation and raw HTML resume creation."""
    system_prompt = """You are an expert ATS Prompt Engineer and Resume Builder.
    Your task is to generate a professional, ATS-optimized, single-page resume in raw semantic HTML/CSS based STRICTLY on the user's data.

    STRICT RULES:
    1. Structure: Standard semantic HTML (<h1> for Name, <h2> for Section Titles, <ul>/<li> for achievements).
    2. Header: Name MUST be on an <h1> tag at the very top, followed by Job Title in <h2>, then contact details & links below.
    3. Zero Data Loss: Do NOT summarize, truncate, or drop technical bullet points, skills, or links.
    4. Formatting & Style: Include an embedded <style> block forcing `body { background-color: #ffffff !important; color: #111111 !important; font-family: 'Helvetica Neue', Arial, sans-serif; padding: 25px; line-height: 1.4; }`. 
    5. Single-Page Constraint: Keep CSS paddings/margins tight so content fits on one A4 page without blank trailing space.
    6. Output: Return ONLY raw HTML code. Do NOT wrap the code in markdown (no ```html).
    """

    user_instructions = f"""
    Generate an ATS-optimized HTML resume for the following user details:

    USER DETAILS:
    {user_query}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_instructions)
    ]

    try:
        response = llm.invoke(messages)
        raw_code = extract_text_from_llm_response(response)
        return clean_html_code(raw_code)
    except Exception as e:
        st.error(f"Failed to generate resume: {str(e)}")
        return ""

def generate_job_cards(llm, location: str, profile: str, tavily_key: str) -> str:
    """Searches live job listings and formats them into an HTML cards layout."""
    search_query = f"{profile} jobs in {location} site:linkedin.com OR site:indeed.com"
    search_results = search_tavily_jobs(search_query, tavily_key)

    prompt = f"""You are a Job Search Assistant.
    Format the following live search results into a clean, modern HTML grid of Job Cards.

    Target Profile: {profile}
    Target Location: {location}
    Search Data: {search_results}

    CRITICAL RULES:
    1. Extract up to 8-10 real, relevant job postings from the search data.
    2. Show: Job Title, Company Name, Location, Estimated Salary (or Market Average), and a Direct Apply button (`<a href="..." target="_blank" class="apply-btn">Apply Now</a>`).
    3. CSS Requirements: Embedded `<style>` block with modern responsive card layout, subtle dark/light border styling, and styled blue action buttons.
    4. Output ONLY raw HTML. Do not wrap in markdown blocks.
    """

    messages = [HumanMessage(content=prompt)]

    try:
        response = llm.invoke(messages)
        raw_code = extract_text_from_llm_response(response)
        return clean_html_code(raw_code)
    except Exception as e:
        st.error(f"Failed to fetch job cards: {str(e)}")
        return ""

st.markdown('<div class="main-header">⚡ AI Resume Generator & Live Job Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Build ATS-Friendly HTML Resumes & Search Real-Time Job Openings using Gemini & Tavily AI</div>', unsafe_allow_html=True)

# Image / Banner Fallback
if os.path.exists("bg.png"):
    st.image("bg.png", use_container_width=True)
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border-radius: 10px; padding: 20px; text-align: center; color: #E2E8F0; margin-bottom: 20px; border: 1px solid #334155;">
        <h4>🚀 Automated Career Assistant</h4>
        <p style="margin:0; font-size: 0.9rem; color: #94A3B8;">Generate semantic resumes, parse existing documents, and discover targeted postings.</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📄 Resume Generator", "💼 Job Finder Agent", "📷 OCR Text Scanner"])

with tab1:
    st.subheader("Generate ATS-Optimized HTML Resume")
    
    default_query = """Agent Persona: Prince Gulia. Profile: Backend Developer and final-year BCA student (2024-2027, 9.52 CGPA) at Guru Gobind Singh Indraprastha University, New Delhi.
Contact: princegulia170306@gmail.com, 8527875112.
Links: github.com/Prince-Gulia-, linkedin.com/in/princegulia, prince-portfolio-xi.vercel.app.
Core Skills: JavaScript, C++, SQL, Python, Node.js, Express.js, PostgreSQL, pgvector, Supabase, MySQL, Redis, BullMQ, Socket.io, JWT Authentication, Gemini API, RAG Pipelines, Linux, Git.
Key Projects:
1. DocuMind (AI Document Assistant): Architected a RAG pipeline utilizing Gemini API for vector embeddings and pgvector for high-performance semantic search. Engineered an asynchronous background queue using BullMQ and ioredis for PDF processing. Optimized search accuracy with a 500-word text chunking algorithm using boundary overlaps.
2. File Processing API: Built an async file ingestion API for multi-format uploads via Multer, BullMQ, and Upstash Redis. Integrated Sharp for image transformation and Cloudinary for CDN delivery.
3. RealChat: Built a real-time chat backend with Socket.io and JWT authentication. Engineered PostgreSQL schema with B-tree indexing for fast message retrieval.
Objective: To engineer scalable REST APIs, real-time architectures, and production-ready AI-integrated systems."""

    user_profile_query = st.text_area("Enter Profile & Experience Details:", value=default_query, height=220)

    if st.button("🚀 Generate Resume"):
        llm = get_llm_model(gemini_api_key)
        if llm:
            with st.spinner("Generating ATS HTML Resume..."):
                html_resume = generate_resume_html(llm, user_profile_query)
                if html_resume:
                    st.success("Resume Generated Successfully!")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.download_button(
                            label="📥 Download HTML Resume",
                            data=html_resume,
                            file_name="Prince_Gulia_Resume.html",
                            mime="text/html"
                        )
                    with col2:
                        st.subheader("Raw Code View")
                        st.text_area("HTML Code", value=html_resume, height=200)

                    st.markdown("---")
                    st.subheader("Live Preview")
                    st.components.v1.html(html_resume, height=800, scrolling=True)

with tab2:
    st.subheader("Find Targeted Job Openings")
    
    c1, c2 = st.columns(2)
    with c1:
        job_profile = st.text_input("Job Profile / Role", value="Backend Developer, Node.js")
    with c2:
        job_location = st.text_input("Location", value="New Delhi, Delhi")

    if st.button("🔍 Search Jobs"):
        llm = get_llm_model(gemini_api_key)
        if llm:
            with st.spinner("Searching active listings on LinkedIn & Indeed via Tavily..."):
                job_cards_html = generate_job_cards(llm, job_location, job_profile, tavily_api_key)
                if job_cards_html:
                    st.success("Job Listings Retrieved!")
                    st.components.v1.html(job_cards_html, height=700, scrolling=True)

with tab3:
    st.subheader("Extract Text from Resume Image (OCR)")
    st.write("Upload an image of an existing resume (.jpg, .png) to extract profile details using PyTesseract.")

    uploaded_file = st.file_uploader("Upload Image File", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Resume Image", width=400)
            
            if st.button("📷 Perform OCR Extraction"):
                with st.spinner("Extracting text via Tesseract OCR..."):
                    extracted_text = pyt.image_to_string(image)
                    if extracted_text.strip():
                        st.success("Text Extracted Successfully!")
                        st.text_area("Extracted OCR Text", value=extracted_text, height=300)
                    else:
                        st.warning("No readable text found in the uploaded image.")
        except Exception as e:
            st.error(f"OCR Error: {str(e)}. Ensure Tesseract is installed and configured in the sidebar.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #718096; font-size: 0.85rem;'>Engineered for High-Performance Backend Resume & Job Search Automation</div>", unsafe_allow_html=True)
