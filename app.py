#==========LOAD MODULES========================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import langchain
from langchain.agents import create_agent

from tavily import TavilyClient
import pytesseract as pyt 
import streamlit as st
import os
import time
from PIL import Image
import pandas as pd
import numpy as np



# To Show web-app: complete page layout
st.set_page_config(layout="wide")

# To Give Title
st.title("AI RESUME GENERATOR")

st.write("""This app helps user to build customized Professional
Resume with Latest Job apply links""")

st.image("bg.png")

st.sidebar.title("Fill Important Details")
st.sidebar.image("bg.png")



# ========API KEYS============# 
# Step 3 API keys
TAVILY_API_KEY = st.sidebar.text_input("Tavily-API",type = "password")
GROQ_API_KEY = st.sidebar.text_input("Groq-API",type = "password")
GOOGLE_API_KEY = st.sidebar.text_input("Gemini-API",type = "password")

all_API = [TAVILY_API_KEY,GROQ_API_KEY,
           GOOGLE_API_KEY ]
if not all(all_API):
    st.error("Must give API keys")
    st.stop()
elif all(all_API):
    st.success("API KEYS LOADED SUCCESSFULLY")
else:
    st.info("PASS ALL API-KEYS")
    


# ================ MODEL====================
model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash-lite',
    google_api_key = GOOGLE_API_KEY
)

# response = model.invoke("Hello Buddy!")
# response.content[-1]['text']


# ======================TOOLS===============
def search_latest_news_jobs(query):
  """This function helps to fetch latest
  news or jobs related article using
  tavily"""

  client = TavilyClient(
      api_key = TAVILY_API_KEY)
  response = client.search(query)
  return response




# Agent Creation
agent = create_agent(
    model = model,
    tools = [search_latest_news_jobs])

# agent


def main_agent(agent, query):
  """This is main agent, or leader agent
  orchestrate sub agents"""

  # Giving prompt to create detailed prompt
  # for code generation
  prompt = """You are AI assistant and
  below given is a prompt, your
  task is to give detailed prompt for
  this.
  You are a professional Resume generator
  where user will give their personal info,
  you have to create detailed Resume
  for students or professional one,
  it must be with dynamic UI and UX and,
  with advanced CSS Professional Designing
  Make sure to give output in HTML format only
  no markdowns allowed
  """

  response = agent.invoke({'messages':[{'role':'user',
                                        'content':prompt}]})
  detailed_prompt = response['messages'][-1].content[-1]['text']

  # SAVE PROMPT using File Handling

  with open('prompt.txt','w') as f:
    f.write(detailed_prompt)

  user_details = f"""Below Given is a user details
  generate Resume based on that, if not
  given keep: Default Resume: Python Developer
  user details: {query}"""

  final_prompt = prompt + detailed_prompt + user_details

  # CODE GENERATION
  response = agent.invoke({'messages':[{'role':'user',
                                        'content':final_prompt}]})
  code = response['messages'][-1].content[-1]['text']

  return code


# code = main_agent(agent,"ALAN TURING, GEN AI EXPERT")
# from IPython import display as DISPLAY
# DISPLAY.HTML(code)



# Fetch Latest Domain related Jobs using Tavily

def get_jobs(agent,
             Location = "Noida,Delhi",
             Profile = "Data Analysts, AI Engineer"):
  Location = "Noida,Delhi"
  Profile = "Data Analysts, AI Engineer"

  prompt = f"""Based on user given Job profile,
  fetch latest jobs or job apply article
  using Naukri, Linkedin, Indeed, or all popular
  Job apply platforms, Show Results with
  JOB PROFILE NAME, LOCATION, SALARY, COMPANY NAME,
  SHOW jobs only related to given
  {Location} and {Profile}. Output must be in
  Professional HTML Naukri theme cards with Dynamic Design,
  Show atleast Top 10-20 results with direct apply link"""


  response = agent.invoke({'messages':[{'role':'user',
                                          'content':prompt}]})
  code = response['messages'][-1].content[-1]['text']

  return code

# code = get_jobs(agent)
# DISPLAY.HTML(code)

# ==========LOAD MODULES========================
import os
import time
import warnings

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract as pyt
from dotenv import load_dotenv

# LangChain & Agent Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

# Tavily Import
from tavily import TavilyClient

# ──────────── Environment & Warnings ────────────
load_dotenv()
warnings.filterwarnings("ignore")


# ──────────── Page Config ────────────
st.set_page_config(
    page_title="AI Resume Generator & Job Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────── Premium CSS ────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.main-header {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    letter-spacing: -0.5px;
}
.sub-header {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-bottom: 1.8rem;
    font-weight: 400;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 0.6rem 1.4rem;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139,92,246,0.4);
    color: white;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #CBD5E1 !important;
    font-weight: 500;
}

/* ── Cards ── */
.card-box {
    background: linear-gradient(145deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.card-box:hover {
    transform: translateY(-3px);
    border-color: #4F46E5;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
}

/* ── Success / Error messages ── */
.stSuccess {
    border-radius: 10px;
}
.stError {
    border-radius: 10px;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: #475569;
    font-size: 0.82rem;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid #1E293B;
    margin-top: 2rem;
}
.app-footer a {
    color: #60A5FA;
    text-decoration: none;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #8B5CF6 !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────── Sidebar: API Keys ────────────
st.sidebar.title("🔑 API Settings")
st.sidebar.markdown("---")
st.sidebar.info("Configure your API credentials below. Keys are loaded automatically from `.env` if present.")

# Retrieve defaults from secrets → env → empty
def _safe_secret(key: str, fallback: str = "") -> str:
    """Read from st.secrets if available, else env var, else fallback."""
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError, Exception):
        return os.environ.get(key, fallback)

DEFAULT_GEMINI_KEY = _safe_secret("GEMINI_API_KEY")
DEFAULT_GROQ_KEY = _safe_secret("GROQ_API_KEY")
DEFAULT_TAVILY_KEY = _safe_secret("TAVILY_API_KEY")

GOOGLE_API_KEY = st.sidebar.text_input("Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password", placeholder="AIzaSy...")
GROQ_API_KEY = st.sidebar.text_input("Groq API Key", value=DEFAULT_GROQ_KEY, type="password", placeholder="gsk_...")
TAVILY_API_KEY = st.sidebar.text_input("Tavily API Key", value=DEFAULT_TAVILY_KEY, type="password", placeholder="tvly-...")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Advanced")

model_choice = st.sidebar.selectbox(
    "LLM Provider",
    ["Gemini (Google)", "Groq (LLaMA)"],
    index=0,
)

tesseract_path = st.sidebar.text_input(
    "Tesseract OCR Path (Optional)",
    value="",
    placeholder=r"e.g. C:\Program Files\Tesseract-OCR\tesseract.exe",
)
if tesseract_path.strip():
    pyt.pytesseract.tesseract_cmd = tesseract_path.strip()


# ──────────── Validate API Keys ────────────
all_API = [TAVILY_API_KEY, GOOGLE_API_KEY]
if model_choice == "Groq (LLaMA)":
    all_API.append(GROQ_API_KEY)

if not all(all_API):
    st.sidebar.error("⚠️ Must provide all required API keys.")
else:
    st.sidebar.success("✅ API Keys Loaded Successfully")


# ──────────── Helper: Extract Text from Agent Response ────────────
def extract_text_from_response(response) -> str:
    """Safely pull the final text string out of any LangChain agent/model response."""
    # Direct string
    if isinstance(response, str):
        return response

    # Agent dict → messages list
    if isinstance(response, dict):
        if "messages" in response and len(response["messages"]) > 0:
            last_msg = response["messages"][-1]
            return extract_text_from_response(last_msg)
        if "output" in response:
            return str(response["output"])

    # AIMessage / HumanMessage objects
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            return "\n".join(parts)

    return str(response)


def clean_html_code(raw_code: str) -> str:
    """Strip markdown fences from generated HTML blocks."""
    code = raw_code.strip()
    if code.startswith("```html"):
        code = code[7:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


# ──────────── Model Initialization ────────────
def get_model(provider: str):
    """Return the selected ChatModel instance."""
    if provider == "Gemini (Google)":
        if not GOOGLE_API_KEY:
            st.error("⚠️ Gemini API Key is missing.")
            return None
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
            )
        except Exception as e:
            st.error(f"Gemini init error: {e}")
            return None
    else:
        if not GROQ_API_KEY:
            st.error("⚠️ Groq API Key is missing.")
            return None
        try:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=GROQ_API_KEY,
                temperature=0.2,
            )
        except Exception as e:
            st.error(f"Groq init error: {e}")
            return None


# ──────────── TOOLS ────────────
@tool
def search_latest_news_jobs(query: str) -> dict:
    """Fetch latest news, articles, or job listings using Tavily search engine.
    Use this whenever the user asks about jobs, news, or current events."""
    if not TAVILY_API_KEY:
        return {"error": "Tavily API key missing."}
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, search_depth="advanced", max_results=10)
        return response
    except Exception as e:
        return {"error": f"Tavily search failed: {str(e)}"}


# ──────────── Agent Creation ────────────
def build_agent(model):
    """Create a LangChain agent (CompiledStateGraph) with search tool."""
    agent = create_agent(
        model=model,
        tools=[search_latest_news_jobs],
    )
    return agent


# ──────────── Core Function: Resume Generation ────────────
def main_agent(agent, query: str) -> str:
    """Main orchestrator — generates a detailed prompt, then creates
    a professional ATS-optimized HTML resume."""

    # Step 1: Generate a detailed resume-builder prompt
    prompt = """You are an AI assistant and professional Resume Generator.
    Your task is to create a DETAILED system prompt for an HTML Resume code generator.
    The system prompt must instruct the code generator to:

    1. Create a single-page, ATS-optimized resume in pure semantic HTML + embedded CSS.
    2. Use a clean white background (#ffffff), dark text (#111111), professional font stack.
    3. Structure: <h1> for Name, <h2> for section headings, <ul>/<li> for bullet points.
    4. Include ALL user details without summarizing or truncating anything.
    5. Use tight CSS margins/paddings to fit everything on one A4 page.
    6. Output ONLY raw HTML — no markdown, no code fences.
    7. Dynamic, modern, professional UI and UX with advanced CSS styling.
    8. Make it visually impressive while remaining ATS-scannable.

    Generate ONLY the detailed system prompt text. Nothing else."""

    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    detailed_prompt = extract_text_from_response(response)

    # Save prompt for debugging / reference
    try:
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(detailed_prompt)
    except Exception:
        pass  # Non-critical — skip silently

    # Step 2: Generate the actual resume HTML using the detailed prompt + user details
    user_details = f"""Below are the user's details.
    Generate a complete, professional HTML resume based on this data.
    If details are sparse, default to a Python Developer profile.

    USER DETAILS:
    {query}"""

    final_prompt = detailed_prompt + "\n\n" + user_details

    response = agent.invoke(
        {"messages": [{"role": "user", "content": final_prompt}]}
    )
    code = extract_text_from_response(response)
    return clean_html_code(code)


# ──────────── Core Function: Job Finder ────────────
def get_jobs(agent, Location: str = "Noida, Delhi", Profile: str = "AI Engineer") -> str:
    """Searches live job listings via Tavily and formats results as HTML cards."""

    prompt = f"""Based on the user's job profile, fetch the latest jobs and
    job-application articles using your search tool.
    Search across Naukri, LinkedIn, Indeed, and all popular job platforms.

    Show results with:
    - JOB PROFILE / TITLE
    - COMPANY NAME
    - LOCATION
    - ESTIMATED SALARY (or market average)
    - DIRECT APPLY LINK (as a styled button)

    Show ONLY jobs relevant to:
    Location: {Location}
    Profile: {Profile}

    Output must be in professional, modern HTML with:
    - Responsive card grid layout
    - Dark/light themed cards with subtle borders & shadows
    - Blue action buttons for "Apply Now"
    - Embedded <style> block for all CSS
    - Show at least Top 10-20 results
    - NO markdown, NO code fences — raw HTML ONLY."""

    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    code = extract_text_from_response(response)
    return clean_html_code(code)


# ──────────── Core Function: OCR Text Extraction ────────────
def extract_ocr_text(image: Image.Image) -> str:
    """Extract text from a PIL Image using PyTesseract."""
    try:
        text = pyt.image_to_string(image)
        return text
    except Exception as e:
        return f"OCR Error: {str(e)}"


# ══════════════════════════════════════════════════════════════
# ██  MAIN UI
# ══════════════════════════════════════════════════════════════

st.markdown(
    '<div class="main-header">⚡ AI Resume Generator & Live Job Agent</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Build ATS-Friendly HTML Resumes & Search Real-Time Job Openings — powered by Gemini, Groq & Tavily AI</div>',
    unsafe_allow_html=True,
)

# Banner Image
if os.path.exists("bg.png"):
    st.image("bg.png", use_container_width=True)
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
                border-radius: 12px; padding: 24px; text-align: center;
                color: #E2E8F0; margin-bottom: 24px; border: 1px solid #334155;">
        <h4 style="margin:0 0 8px 0;">🚀 Automated Career Assistant</h4>
        <p style="margin:0; font-size: 0.9rem; color: #94A3B8;">
            Generate semantic resumes, parse existing documents, and discover targeted job postings.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ──────────── Tabs ────────────
tab1, tab2, tab3 = st.tabs(["📄 Resume Generator", "💼 Job Finder Agent", "📷 OCR Text Scanner"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Resume Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("✨ Generate ATS-Optimized HTML Resume")
    st.markdown(
        '<p style="color:#94A3B8; font-size:0.9rem;">Paste your profile details below and let the AI agent craft a pixel-perfect resume.</p>',
        unsafe_allow_html=True,
    )

    default_query = """Agent Persona: Prince Gulia. Profile: Backend Developer and final-year BCA student (2024-2027, 9.52 CGPA) at Guru Gobind Singh Indraprastha University, New Delhi.
Contact: princegulia170306@gmail.com, 8527875112.
Links: github.com/Prince-Gulia-, linkedin.com/in/princegulia, prince-portfolio-xi.vercel.app.
Core Skills: JavaScript, C++, SQL, Python, Node.js, Express.js, PostgreSQL, pgvector, Supabase, MySQL, Redis, BullMQ, Socket.io, JWT Authentication, Gemini API, RAG Pipelines, Linux, Git.
Key Projects:
1. DocuMind (AI Document Assistant): Architected a RAG pipeline utilizing Gemini API for vector embeddings and pgvector for high-performance semantic search. Engineered an asynchronous background queue using BullMQ and ioredis for PDF processing. Optimized search accuracy with a 500-word text chunking algorithm using boundary overlaps.
2. File Processing API: Built an async file ingestion API for multi-format uploads via Multer, BullMQ, and Upstash Redis. Integrated Sharp for image transformation and Cloudinary for CDN delivery.
3. RealChat: Built a real-time chat backend with Socket.io and JWT authentication. Engineered PostgreSQL schema with B-tree indexing for fast message retrieval.
Objective: To engineer scalable REST APIs, real-time architectures, and production-ready AI-integrated systems."""

    user_profile_query = st.text_area(
        "Enter Profile & Experience Details:",
        value=default_query,
        height=220,
        placeholder="Paste your complete profile, skills, projects, education...",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        generate_btn = st.button("🚀 Generate Resume", use_container_width=True)

    if generate_btn:
        if not all(all_API):
            st.error("⚠️ Please provide all required API keys in the sidebar.")
        else:
            model = get_model(model_choice)
            if model:
                agent = build_agent(model)
                with st.spinner("🔄 AI Agent is crafting your resume — this may take 30-60 seconds..."):
                    start_time = time.time()
                    html_resume = main_agent(agent, user_profile_query)
                    elapsed = round(time.time() - start_time, 1)

                if html_resume:
                    st.success(f"✅ Resume Generated Successfully! ({elapsed}s)")

                    # Store in session state so it survives interactions
                    st.session_state["resume_html"] = html_resume

                    # Action buttons row
                    act1, act2, act3 = st.columns([1, 1, 2])
                    with act1:
                        st.download_button(
                            label="📥 Download HTML",
                            data=html_resume,
                            file_name="Resume.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                    with act2:
                        st.download_button(
                            label="📋 Download Code",
                            data=html_resume,
                            file_name="Resume_Code.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

                    # Code viewer
                    with st.expander("🔍 View Raw HTML Code", expanded=False):
                        st.code(html_resume, language="html")

                    # Live Preview
                    st.markdown("---")
                    st.subheader("👁️ Live Preview")
                    st.components.v1.html(html_resume, height=900, scrolling=True)
                else:
                    st.error("❌ Resume generation returned empty. Try again.")

    # Persist previous result across tab switches
    elif "resume_html" in st.session_state and st.session_state["resume_html"]:
        st.info("ℹ️ Showing previously generated resume. Click **Generate Resume** to create a new one.")
        html_resume = st.session_state["resume_html"]

        act1, act2, _ = st.columns([1, 1, 2])
        with act1:
            st.download_button(
                label="📥 Download HTML",
                data=html_resume,
                file_name="Resume.html",
                mime="text/html",
                use_container_width=True,
            )
        with act2:
            st.download_button(
                label="📋 Download Code",
                data=html_resume,
                file_name="Resume_Code.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with st.expander("🔍 View Raw HTML Code", expanded=False):
            st.code(html_resume, language="html")
        st.markdown("---")
        st.subheader("👁️ Live Preview")
        st.components.v1.html(html_resume, height=900, scrolling=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Job Finder Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("🔎 Find Targeted Job Openings")
    st.markdown(
        '<p style="color:#94A3B8; font-size:0.9rem;">The AI agent searches LinkedIn, Indeed, Naukri & more via Tavily to fetch real-time job listings.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        job_profile = st.text_input(
            "Job Profile / Role",
            value="Backend Developer, Node.js",
            placeholder="e.g. Data Analyst, ML Engineer...",
        )
    with c2:
        job_location = st.text_input(
            "Preferred Location",
            value="New Delhi, Delhi",
            placeholder="e.g. Bangalore, Remote...",
        )

    col_j1, col_j2 = st.columns([1, 4])
    with col_j1:
        search_btn = st.button("🔍 Search Jobs", use_container_width=True)

    if search_btn:
        if not all(all_API):
            st.error("⚠️ Please provide all required API keys in the sidebar.")
        else:
            model = get_model(model_choice)
            if model:
                agent = build_agent(model)
                with st.spinner("🌐 Searching active listings across job platforms..."):
                    start_time = time.time()
                    job_cards_html = get_jobs(agent, Location=job_location, Profile=job_profile)
                    elapsed = round(time.time() - start_time, 1)

                if job_cards_html:
                    st.success(f"✅ Job Listings Retrieved! ({elapsed}s)")
                    st.session_state["job_html"] = job_cards_html
                    st.components.v1.html(job_cards_html, height=800, scrolling=True)
                else:
                    st.error("❌ No job listings returned. Try a different query.")

    elif "job_html" in st.session_state and st.session_state["job_html"]:
        st.info("ℹ️ Showing previous job search results. Click **Search Jobs** to refresh.")
        st.components.v1.html(st.session_state["job_html"], height=800, scrolling=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: OCR Text Scanner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("📷 Extract Text from Resume Image (OCR)")
    st.markdown(
        '<p style="color:#94A3B8; font-size:0.9rem;">Upload a photo of an existing resume (.jpg, .png) to extract text using Tesseract OCR. You can then paste it into the Resume Generator.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Resume Image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            img_col, info_col = st.columns([2, 1])
            with img_col:
                st.image(image, caption="📎 Uploaded Resume Image", width=450)
            with info_col:
                st.markdown(f"""
                <div class="card-box">
                    <p><strong>📁 Filename:</strong> {uploaded_file.name}</p>
                    <p><strong>📐 Dimensions:</strong> {image.size[0]} × {image.size[1]} px</p>
                    <p><strong>💾 Size:</strong> {round(uploaded_file.size / 1024, 1)} KB</p>
                </div>
                """, unsafe_allow_html=True)

            ocr_btn = st.button("📷 Perform OCR Extraction", use_container_width=False)

            if ocr_btn:
                with st.spinner("🔍 Extracting text via Tesseract OCR..."):
                    extracted_text = extract_ocr_text(image)

                if extracted_text.strip() and not extracted_text.startswith("OCR Error"):
                    st.success("✅ Text Extracted Successfully!")
                    st.text_area(
                        "Extracted OCR Text",
                        value=extracted_text,
                        height=350,
                        help="Copy this text and paste it into the Resume Generator tab.",
                    )
                    st.download_button(
                        label="📥 Download Extracted Text",
                        data=extracted_text,
                        file_name="extracted_resume_text.txt",
                        mime="text/plain",
                    )
                elif extracted_text.startswith("OCR Error"):
                    st.error(f"❌ {extracted_text}")
                    st.info("💡 Make sure Tesseract is installed and configured in the sidebar.")
                else:
                    st.warning("⚠️ No readable text found in the uploaded image. Try a clearer image.")

        except Exception as e:
            st.error(f"❌ Image Error: {str(e)}")
            st.info("💡 Ensure Tesseract OCR is installed. Set the path in the sidebar if needed.")


# ──────────── Footer ────────────
st.markdown("---")
st.markdown("""
<div class="app-footer">
    ⚡ Engineered for High-Performance Resume & Job Search Automation<br>
    Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> •
    <a href="https://ai.google.dev" target="_blank">Gemini</a> •
    <a href="https://tavily.com" target="_blank">Tavily</a> •
    <a href="https://groq.com" target="_blank">Groq</a>
</div>
""", unsafe_allow_html=True)

