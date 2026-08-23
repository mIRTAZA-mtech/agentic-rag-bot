import os
import datetime
import requests
import wikipedia
from zoneinfo import ZoneInfo
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

import streamlit as st

# LangChain core
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Groq wrapper
from langchain_groq import ChatGroq

# LangGraph
from langgraph.prebuilt import create_react_agent

# ----------------------------------------------------------------
# Load environment
# ----------------------------------------------------------------
_ = load_dotenv(find_dotenv())

# ----------------------------------------------------------------
# Streamlit Page Config & Custom CSS
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AI Agent",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ── Header ──
st.markdown("""
<div class="hero-header">
    <div class="logo">◉</div>
    <h1>AI Agent</h1>
    <p>Weather · Wikipedia · Converter · World Clock · PDF Q&A</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Capability Pills ──
st.markdown("""
<div class="pills">
    <span class="pill">🌡 Weather</span>
    <span class="pill">📖 Wikipedia</span>
    <span class="pill">💱 Converter</span>
    <span class="pill">🕐 World Clock</span>
    <span class="pill">📄 PDF Q&A</span>
</div>
""", unsafe_allow_html=True)


# ================================================================
#  TOOLS  (unchanged backend logic)
# ================================================================

# ----------------------------------------------------------------
# Tool 1: Weather Tool
# ----------------------------------------------------------------
class OpenMeteoInput(BaseModel):
    """Input schema for the Open-Meteo weather API."""
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature using Open-Meteo API for given coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return "Error: Unable to fetch temperature data"

    data = response.json()
    now = datetime.datetime.now()

    # Find the closest hourly reading to the current time
    times = [datetime.datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    temperatures = data["hourly"]["temperature_2m"]

    closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - now))
    temp = temperatures[closest_idx]

    return f"The current temperature at ({latitude}, {longitude}) is {temp}°C"

# ----------------------------------------------------------------
# Tool 2: Wikipedia Search
# ----------------------------------------------------------------
@tool
def search_wikipedia(query: str) -> str:
    """Use wikipedia to answer questions that require general world knowledge"""
    try:
        title = wikipedia.search(query, results=5)
    except Exception as e:
        return f"Wikipedia search failed: {e}"
        
    summaries = []
    for t in title[:3]:
        try:
            page = wikipedia.page(t, auto_suggest=False)
            summaries.append(f"title: {page.title}\ncontent: {page.content[:1000]}\nurl: {page.url}")
        except Exception:
            pass
    return "\n\n".join(summaries)

# ----------------------------------------------------------------
# Tool 3: Currency & Unit Converter
# ----------------------------------------------------------------
UNIT_CONVERSIONS = {
    ("km", "miles"):  0.621371,
    ("miles", "km"):  1.60934,
    ("kg", "lbs"):    2.20462,
    ("lbs", "kg"):    0.453592,
    ("m", "ft"):      3.28084,
    ("ft", "m"):      0.3048,
    ("l", "gal"):     0.264172,
    ("gal", "l"):     3.78541,
    ("cm", "in"):     0.393701,
    ("in", "cm"):     2.54,
}

def _convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Handle temperature conversions separately."""
    f, t = from_unit.lower(), to_unit.lower()
    if f == "c" and t == "f":
        return f"{value} C = {value * 9/5 + 32:.2f} F"
    elif f == "f" and t == "c":
        return f"{value} F = {(value - 32) * 5/9:.2f} C"
    elif f == "c" and t == "k":
        return f"{value} C = {value + 273.15:.2f} K"
    elif f == "k" and t == "c":
        return f"{value} K = {value - 273.15:.2f} C"
    elif f == "f" and t == "k":
        return f"{value} F = {(value - 32) * 5/9 + 273.15:.2f} K"
    elif f == "k" and t == "f":
        return f"{value} K = {(value - 273.15) * 9/5 + 32:.2f} F"
    return None

class ConvertInput(BaseModel):
    """Input schema for currency and unit conversions."""
    value: float = Field(..., description="The numeric value to convert")
    from_unit: str = Field(..., description="Source unit or 3-letter currency code, e.g. 'km', 'kg', 'USD', 'C'")
    to_unit: str = Field(..., description="Target unit or 3-letter currency code, e.g. 'miles', 'lbs', 'EUR', 'F'")

@tool(args_schema=ConvertInput)
def convert_currency_or_unit(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between currencies (USD, EUR, PKR, etc.) using live exchange rates,
    or between common units (km/miles, kg/lbs, C/F, m/ft, L/gal, cm/in)."""

    f = from_unit.strip().lower()
    t = to_unit.strip().lower()

    # Temperature
    temp_result = _convert_temperature(value, f, t)
    if temp_result:
        return temp_result

    # Unit conversion
    key = (f, t)
    if key in UNIT_CONVERSIONS:
        result = value * UNIT_CONVERSIONS[key]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # Currency conversion
    try:
        url = f"https://open.er-api.com/v6/latest/{f.upper()}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Error: Could not fetch exchange rates (HTTP {response.status_code})"

        data = response.json()
        if data.get("result") != "success":
            return f"Error: API returned {data.get('result', 'unknown error')}"

        rate = data["rates"].get(t.upper())
        if rate is None:
            return f"Error: Unknown currency code '{to_unit.upper()}'"

        converted = value * rate
        return f"{value} {f.upper()} = {converted:.2f} {t.upper()} (rate: {rate})"

    except requests.RequestException as e:
        return f"Error fetching exchange rate: {e}"

# ----------------------------------------------------------------
# Tool 4: World Clock / Time Zone
# ----------------------------------------------------------------
CITY_TO_TIMEZONE = {
    "tokyo": "Asia/Tokyo", "new york": "America/New_York", "london": "Europe/London",
    "paris": "Europe/Paris", "berlin": "Europe/Berlin", "sydney": "Australia/Sydney",
    "dubai": "Asia/Dubai", "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "karachi": "Asia/Karachi",
    "lahore": "Asia/Karachi", "islamabad": "Asia/Karachi", "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai", "moscow": "Europe/Moscow", "istanbul": "Europe/Istanbul",
    "cairo": "Africa/Cairo", "los angeles": "America/Los_Angeles", "chicago": "America/Chicago",
    "toronto": "America/Toronto", "sao paulo": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires", "riyadh": "Asia/Riyadh",
    "jakarta": "Asia/Jakarta", "seoul": "Asia/Seoul", "bangkok": "Asia/Bangkok",
    "nairobi": "Africa/Nairobi", "johannesburg": "Africa/Johannesburg",
    "amsterdam": "Europe/Amsterdam", "rome": "Europe/Rome", "madrid": "Europe/Madrid",
    "lisbon": "Europe/Lisbon", "zurich": "Europe/Zurich", "vienna": "Europe/Vienna",
    "warsaw": "Europe/Warsaw", "kuala lumpur": "Asia/Kuala_Lumpur", "auckland": "Pacific/Auckland",
    "honolulu": "Pacific/Honolulu", "denver": "America/Denver", "vancouver": "America/Vancouver",
    "mexico city": "America/Mexico_City", "lima": "America/Lima", "bogota": "America/Bogota",
    "doha": "Asia/Qatar", "dhaka": "Asia/Dhaka", "kathmandu": "Asia/Kathmandu",
    "colombo": "Asia/Colombo", "lagos": "Africa/Lagos", "accra": "Africa/Accra",
    "casablanca": "Africa/Casablanca",
}

@tool
def get_current_time(location: str) -> str:
    """Get the current date and time for any city or IANA timezone."""
    location_clean = location.strip().lower()
    tz_name = CITY_TO_TIMEZONE.get(location_clean)
    if not tz_name:
        tz_name = location.strip()

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return f"Unknown location or timezone: '{location}'."

    now = datetime.datetime.now(tz)
    return (
        f"Current time in {location}:\n"
        f"  Date:  {now.strftime('%A, %B %d, %Y')}\n"
        f"  Time:  {now.strftime('%I:%M %p')}\n"
        f"  Zone:  {tz_name} (UTC{now.strftime('%z')})"
    )

# ----------------------------------------------------------------
# Tool 5: PDF Knowledge Base Q&A (RAG)
# ----------------------------------------------------------------
PDF_DOCS_DIRECTORY = os.getenv("PDF_DOCS_DIRECTORY", "./docs")
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "1000"))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "150"))
PDF_RETRIEVAL_K = int(os.getenv("PDF_RETRIEVAL_K", "4"))
PDF_QA_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

@st.cache_resource
def get_pdf_vectorstore():
    """Build and return a Chroma vector store over local PDFs."""
    if not os.path.isdir(PDF_DOCS_DIRECTORY):
        return None
        
    pdfs = [os.path.join(PDF_DOCS_DIRECTORY, f) for f in os.listdir(PDF_DOCS_DIRECTORY) if f.endswith(".pdf")]
    if not pdfs:
        return None

    # Lazy import to avoid loading HuggingFace models unnecessarily
    from langchain_huggingface import HuggingFaceEmbeddings

    documents = []
    for path in pdfs:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=PDF_CHUNK_SIZE, chunk_overlap=PDF_CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

class PdfQAInput(BaseModel):
    """Input schema for the PDF knowledge-base Q&A tool."""
    question: str = Field(..., description="The question to answer using the local PDF documents")

@tool(args_schema=PdfQAInput)
def query_pdf_documents(question: str) -> str:
    """Answer a question using the PDF documents found in the local docs directory."""
    vectorstore = get_pdf_vectorstore()
    if not vectorstore:
        return "No local PDF documents found or directory does not exist."

    retriever = vectorstore.as_retriever(search_kwargs={"k": PDF_RETRIEVAL_K})
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "No relevant information found in the PDF documents."

    context = "\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}, "
        f"page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in relevant_docs
    )

    llm = ChatGroq(model=PDF_QA_MODEL, temperature=0)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't contained in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response.content


# ================================================================
#  AGENT SETUP
# ================================================================

tools = [
    get_current_temperature,
    search_wikipedia,
    convert_currency_or_unit,
    get_current_time,
    query_pdf_documents,
]

# Ensure API Key is available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.markdown(
        '<div class="status-toast">⚠ GROQ_API_KEY missing — add it to your .env file.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Initialize the Groq model and LangGraph Agent
llm = ChatGroq(api_key=groq_api_key, model=PDF_QA_MODEL, temperature=0.2)
agent_executor = create_react_agent(llm, tools)


# ================================================================
#  CHAT UI
# ================================================================

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Empty state (shown when no messages yet)
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">💬</div>
        <p>Ask me about the weather, look something up on Wikipedia,
        convert currencies &amp; units, check world clocks, or upload a PDF to chat with it.</p>
    </div>
    """, unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Message…", accept_file=True, file_type=["pdf"]):
    # Handle string vs ChatInputValue
    if isinstance(prompt, str):
        user_input = prompt
        uploaded_files = []
    else:
        user_input = getattr(prompt, "text", "")
        uploaded_files = getattr(prompt, "files", [])

    # Process any uploaded files first
    if uploaded_files:
        if not os.path.exists(PDF_DOCS_DIRECTORY):
            os.makedirs(PDF_DOCS_DIRECTORY)
            
        new_files_added = False
        file_names = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(PDF_DOCS_DIRECTORY, uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_added = True
            file_names.append(uploaded_file.name)
                
        if new_files_added:
            get_pdf_vectorstore.clear()  # Clear cache to rebuild the vector database
            
        # Add a status message to chat history
        names_str = ", ".join(file_names)
        status_msg = f"📄 Uploaded **{names_str}** — ready to answer questions about it."
        st.session_state.messages.append({"role": "assistant", "content": status_msg})
        with st.chat_message("assistant"):
            st.markdown(status_msg)

    # Process text input
    if user_input:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Convert session messages to LangGraph expected format
        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history.append(("human", msg["content"]))
            else:
                chat_history.append(("assistant", msg["content"]))

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    # Invoke the agent
                    result = agent_executor.invoke({"messages": chat_history})
                    
                    # Get the final response (the last message in the list)
                    final_response = result["messages"][-1].content
                    
                    st.markdown(final_response)
                    # Save assistant response to state
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                except Exception as e:
                    st.error(f"Something went wrong: {e}")