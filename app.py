from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import langchain
from langchain.agents import create_agent
import langchain_community
from tavily import TavilyClient
import pytesseract as pyt
import os
import time
from PIL import Image
import pandas as pd
import numpy as np
import warnings

# To Show web-app: complete page layout
st.set_page_config(layout="wide")

# To Give Title
st.title("AI RESUME GENERATOR")

st.write("""This app helps user to build customized Professional
Resume with Latest Job apply links""")

st.image("bg.png")

# API KEYS
GEMINI_API_KEY = "AQ.Ab8RN6Jkv5p04YWXvhp5SzkVFgbKK8XxbCTu77udtWfTUYz-8Q"
GROQ_API_KEY = "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"
TAVILY_API_KEY="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"

# Model Creation
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key = GEMINI_API_KEY
)

# response = model.invoke("Hello Brother")
# response.content[-1]['text']

# Tool for getting latest news related Job
def search_latest_news_jobs(query):
  """This function helps to fetch the latest news and jobs related to the query using tavily"""

  client = TavilyClient(
      api_key = TAVILY_API_KEY
  )
  response = client.search(query)
  return response

# Creating a model
agent = create_agent(
    model=model,
    tools = [search_latest_news_jobs]
)
# agent

# A tool for handling all the agents, main agent that gives the detailed orders and instructions
from IPython.display import display, HTML

def main_agent(agent, query):
    """This is the main agent that orchestrates the resume generation."""

    # 1. FIXED PROMPT: Added explicit CSS color rendering rules
    prompt = """You are an expert AI prompt engineer. Your task is to write a highly detailed system prompt for an advanced Resume Generator Agent.

    The Resume Generator Agent must create professional, ATS-optimized resumes in pure HTML/CSS.
    Crucial rules the generated prompt MUST enforce on the agent:
    1. ATS-Friendly Structure: Use standard, semantic HTML (<h1> for Name, <h2> for section headers, <ul>/<li> for bullet points). Avoid complex CSS grids or flexbox layouts for text-heavy sections that confuse ATS parsers.
    2. Header Layout: The Name MUST be on its own line at the very top (<h1>), followed immediately by the Job Title (<h2>) on the next line. Include all contact info and links clearly below it.
    3. Zero Data Loss: The agent MUST NOT summarize, shorten, or remove technical details, metrics, keywords, or bullet points provided by the user.
    4. Single Page Constraint: Use compact CSS (adjusted padding, margins, line-height, and professional fonts) to ensure the resume fits cleanly on one page. Prevent any blank second pages.
    5. Output Format: Return ONLY raw HTML code. Do not wrap the output in markdown blocks (e.g., no ```html).
    6. Visibility & Styling (CRITICAL): The HTML MUST include an embedded <style> block that explicitly sets `body { background-color: #ffffff !important; color: #111111 !important; }`. All headings, links, and text must be explicitly colored black or dark gray to ensure high contrast and perfect readability, even when rendered in dark-mode IDEs. Do not use dark backgrounds.
    """

    # Generate the detailed prompt based on the new ATS-focused rules
    response = agent.invoke({"messages" : [{"role" : "user", "content" : prompt}]})
    detailed_prompt = response['messages'][-1].content[-1]['text']

    with open('prompt.txt', 'w') as f:
        f.write(detailed_prompt)

    # 2. FIXED USER INSTRUCTIONS: Demand strict adherence to the data
    user_details = f"""
    Below is the user's profile and data.
    Generate a highly professional, ATS-optimized HTML resume based STRICTLY on this data. Do not omit any technical bullet points, skills, or links.

    USER DETAILS: {query}
    """

    final_prompt = detailed_prompt + "\n\n" + user_details

    # 3. CODE GENERATION
    response = agent.invoke({'messages': [{'role':'user', 'content':final_prompt}]})
    code = response['messages'][-1].content[-1]['text']

    # 4. Cleanup: Strip markdown wrappers just in case the LLM disobeys the format rule
    code = code.replace("```html", "").replace("```", "").strip()

    return code

# 5. The highly-detailed query string
query_string = """
Agent Persona: Prince Gulia. Profile: Backend Developer and final-year BCA student (2024-2027, 9.52 CGPA) at Guru Gobind Singh Indraprastha University, New Delhi.
Contact: princegulial70306@gmail.com, 8527875112.
Links: [github.com/Prince-Gulia-](https://github.com/Prince-Gulia-), [linkedin.com/in/princegulia](https://linkedin.com/in/princegulia), prince-portfolio-xi.vercel.app.
Core Skills: JavaScript, C++, SQL, Python, Node.js, Express.js, PostgreSQL, pgvector, Supabase, MySQL, SQLite, Redis, BullMQ, Socket.io, JWT Authentication, Gemini API, RAG Pipelines, Multer, Cloudinary, Sharp, Git, Linux.
Key Projects:
1. DocuMind (AI Document Assistant): Architected a RAG pipeline using Gemini API for vector embeddings and pgvector for cosine similarity search. Engineered an asynchronous PDF processing queue using BullMQ and ioredis. Improved retrieval quality with a custom 500-word boundary overlap chunking strategy.
2. File Processing API: Built an async file ingestion API for multi-format uploads via Multer, BullMQ, and Upstash Redis. Integrated Sharp for server-side image transformation and Cloudinary for CDN delivery. Configured exponential backoff retry logic.
3. RealChat: Built a real-time chat backend with Socket.io and JWT authentication. Engineered a PostgreSQL schema with B-tree indexing to speed up chat history retrieval. Implemented secure room-based broadcasting.
Objective: To engineer scalable REST APIs, real-time architectures, and production-ready AI-integrated systems.
"""

# 6. Execute and render
# html_resume = main_agent(agent, query_string)
# display(HTML(html_resume))

# Calling the function
# from IPython.display import display, HTML

# 3. Call your function
query_string = "Agent Persona: Prince Gulia. Profile: Backend Developer and final-year BCA student (2024-2027, 9.52 CGPA) at Guru Gobind Singh Indraprastha University, New Delhi. Contact: princegulial70306@gmail.com, 8527875112, github.com/Prince-Gulia-. Core Skills: JavaScript, C++, SQL, Python, PHP, Node.js, Express, PostgreSQL, pgvector, Redis, BullMQ, Socket.io, Gemini API, RAG Pipelines. Key Projects: DocuMind (AI Document Assistant with Gemini API and async processing), File Processing API (multi-format background processing), RealChat (real-time WebSockets), FairPrice (price intelligence platform), DataLens (dataset analyzer), and a full-stack Study Tracker. Traits & Philosophy: Highly focused on backend efficiency, system scalability, and digging into complex, hidden programming concepts. Values hard work, continuous learning, and peace; actively avoids passive consumption and scrolling. Environment Preferences: Works best in dark-themed coding environments (specifically shades like 'Betel Leaf') while listening to Punjabi, Haryanvi, Electronic, and Phonk music. Objective: To engineer scalable REST APIs, real-time architectures, and production-ready AI-integrated systems."

# code = main_agent(agent, query_string)

# 4. Wrap the HTML object inside the display() function to render it
# display(HTML(code))

# Tool for getting jobs
def get_jobs(agent, Location="New Delhi, Delhi", Profile="Backend Developer, Node.JS"):

    # Enhanced prompt instructing the agent to specifically search LinkedIn and Indeed
    prompt = f"""You are an expert job search assistant and web developer.
    Your task is to find real, active job postings for the role of '{Profile}' in '{Location}'.

    CRITICAL SEARCH CONSTRAINTS:
    1. ONLY source job postings from Indeed and LinkedIn. Do not use Naukri or other platforms.
    2. Do NOT hallucinate or create fake job listings.
    3. The 'Direct Apply Link' MUST be the actual, working HTTPS URL to the specific job posting on Indeed or LinkedIn. Do not use placeholders like '#' or '#apply'.

    Requirements for Output:
    1. Show 10 highly relevant job results.
    2. Data points to include for each job card:
        - JOB PROFILE NAME
        - LOCATION
        - SALARY (Provide the listed salary or an estimate based on market averages)
        - COMPANY NAME
        - Direct Apply Link (Must be a functional URL opening in a new tab: target="_blank")
    3. UI/UX Design Requirements:
        - Output MUST be strictly in HTML format. No markdown blocks (` ```html `).
        - Use a professional, dynamic design inspired by modern job portals (clean white backgrounds, subtle shadow borders, professional blue accent colors for links/buttons).
        - Display the results as modern, responsive HTML/CSS cards.
        - The Apply button must be fully clickable and route to the actual LinkedIn/Indeed URL.
    """

    response = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})

    # Extract the text content from the response
    code = response['messages'][-1].content[-1]['text']

    # Clean up potential markdown formatting if the agent disobeys the raw HTML instruction
    code = code.replace("```html\n", "").replace("```html", "").replace("```", "").strip()

    return code

# code = get_jobs(agent)
# display(HTML(code)) 
