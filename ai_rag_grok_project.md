# 🤖 AI Conversational Agent — Powered by Groq & LangGraph

> A multi-tool, agentic AI assistant with a full RAG pipeline, built from scratch using LangGraph, Groq, Streamlit, and HuggingFace embeddings.

---

## 📌 Project Overview

This project is a **production-grade conversational AI agent** that goes far beyond a simple chatbot. It uses an **agentic architecture** — meaning the AI doesn't just generate text, it *reasons about which tools to use*, executes them autonomously, and synthesizes the results into a coherent response.

The agent is powered by **Groq's ultra-fast LLM inference** (Llama 3.1 70B) and orchestrated through **LangGraph's ReAct agent framework**. It runs on a clean **Streamlit** web interface with a persistent chat experience.

**What makes this special:** The agent autonomously decides — based on the user's question — whether to fetch live weather data, look up Wikipedia, convert currencies, check world time zones, or search through uploaded PDF documents using a full **Retrieval-Augmented Generation (RAG)** pipeline. No manual tool selection required.

---

## 🎯 Problem Statement

Most chatbots are limited to static, pre-trained knowledge. They can't:
- Access **real-time data** (weather, exchange rates, current time)
- Reason over **your private documents** (PDFs, reports, manuals)
- **Decide** which data source to use based on context

This project solves all three problems by building an **agentic system** where the LLM acts as a reasoning engine that dynamically selects and invokes the right tool for each query.

---

## 🔭 Project Scope

### What It Does
| Capability | Description |
|---|---|
| **Multi-tool AI Agent** | Autonomously selects from 5 integrated tools based on user intent |
| **RAG over PDFs** | Upload any PDF and ask questions — the agent retrieves relevant chunks and generates grounded answers |
| **Live Weather** | Real-time temperature data from Open-Meteo API for any coordinates on Earth |
| **Wikipedia Search** | Instant access to world knowledge via Wikipedia's API |
| **Currency & Unit Conversion** | Live exchange rates + common unit conversions (km↔miles, kg↔lbs, °C↔°F, etc.) |
| **World Clock** | Current time for 50+ major cities worldwide |
| **Conversational Memory** | Full chat history maintained across the session via Streamlit session state |
| **Inline PDF Upload** | Drag-and-drop PDF upload directly in the chat bar (no sidebar needed) |

### What It Demonstrates
- **Agentic AI design patterns** — the ReAct (Reason + Act) loop
- **Tool-use with LLMs** — structured tool calling via Pydantic schemas
- **RAG pipeline engineering** — document loading, chunking, embedding, vector storage, retrieval, and grounded generation
- **API integration** — working with multiple external APIs (weather, exchange rates, Wikipedia)
- **Full-stack AI application** — from backend agent logic to frontend UI

---

## 🏗️ Technical Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Chat Input  │  │ PDF Upload   │  │ Chat History  │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────────┘  │
│         │                │                              │
├─────────┼────────────────┼──────────────────────────────┤
│         ▼                ▼                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │           LangGraph ReAct Agent                  │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │         Groq LLM (Llama 3.1 70B)        │    │    │
│  │  │      Ultra-fast inference (~300 tok/s)   │    │    │
│  │  └─────────────┬───────────────────────────┘    │    │
│  │                │                                │    │
│  │    ┌───────────┼───────────────────┐            │    │
│  │    ▼           ▼           ▼       ▼       ▼    │    │
│  │ ┌──────┐ ┌──────────┐ ┌───────┐ ┌─────┐ ┌───┐ │    │
│  │ │Weather│ │Wikipedia │ │Convert│ │Clock│ │RAG│ │    │
│  │ │ Tool  │ │  Tool    │ │ Tool  │ │Tool │ │Tool│ │    │
│  │ └──┬───┘ └────┬─────┘ └───┬───┘ └──┬──┘ └─┬─┘ │    │
│  │    │          │           │        │      │    │    │
│  └────┼──────────┼───────────┼────────┼──────┼────┘    │
│       ▼          ▼           ▼        ▼      ▼         │
│  ┌─────────┐ ┌────────┐ ┌────────┐ ┌────┐ ┌────────┐  │
│  │Open-Meteo│ │Wikipedia│ │ExchRate│ │Zone│ │ChromaDB│  │
│  │   API    │ │  API    │ │  API   │ │Info│ │+HFEmbed│  │
│  └─────────┘ └────────┘ └────────┘ └────┘ └────────┘  │
└─────────────────────────────────────────────────────────┘
```

### The ReAct Agent Loop

The core of this project is the **ReAct (Reason + Act) pattern**, implemented via LangGraph's `create_react_agent`:

1. **User sends a message** → the full chat history is passed to the LLM
2. **LLM reasons** about what tools (if any) are needed
3. **LLM generates a tool call** with structured arguments (validated via Pydantic)
4. **Tool executes** and returns results
5. **LLM synthesizes** the tool output into a natural language response
6. If the LLM determines more tools are needed, it loops back to step 2 (multi-hop reasoning)

This is fundamentally different from a simple prompt→response chatbot. The agent can chain multiple tool calls, handle ambiguous queries, and self-correct.

---

## 🧰 Tech Stack Breakdown

| Layer | Technology | Why This Choice |
|---|---|---|
| **LLM Provider** | [Groq](https://groq.com) | Fastest inference available — ~300 tokens/second on Llama 3.1 70B. Free tier available. Uses custom LPU hardware. |
| **LLM Model** | Llama 3.1 70B (via Groq) | Open-source, strong tool-calling capability, excellent instruction following |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) | Production-grade agent orchestration. ReAct agent with built-in tool routing, retry logic, and state management |
| **Frontend** | [Streamlit](https://streamlit.io) | Rapid prototyping of data/AI apps. Built-in chat UI, session state, file upload, caching |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Lightweight (80MB), runs locally, no API costs. 384-dimensional sentence embeddings |
| **Vector Store** | [ChromaDB](https://www.trychroma.com) | Lightweight, in-memory vector database. Zero config, perfect for local RAG |
| **Document Loader** | LangChain `PyPDFLoader` | Extracts text from PDF files page-by-page with metadata |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Intelligent chunking that respects sentence/paragraph boundaries |
| **Data Validation** | Pydantic | Type-safe tool input schemas — the LLM generates structured JSON that is validated before tool execution |

---

## 🔧 The Five Integrated Tools (Deep Dive)

### 1. 🌤️ Live Weather — `get_current_temperature`
- **API**: [Open-Meteo](https://open-meteo.com) (free, no API key required)
- **Input**: Latitude & longitude (the LLM resolves city names to coordinates)
- **How it works**: Fetches hourly forecast data, finds the reading closest to the current time
- **Output**: Current temperature in Celsius

### 2. 📚 Wikipedia Search — `search_wikipedia`
- **API**: Python `wikipedia` library
- **Input**: Natural language query
- **How it works**: Searches for top 5 results, fetches the first 3 page summaries (1000 chars each), returns them with URLs
- **Use case**: General world knowledge, historical facts, definitions, biographies

### 3. 💱 Currency & Unit Converter — `convert_currency_or_unit`
- **APIs**: [ExchangeRate-API](https://open.er-api.com) for currencies + built-in conversion tables for units
- **Supports**:
  - **Currencies**: Any world currency (USD, EUR, PKR, GBP, etc.) via live exchange rates
  - **Units**: km↔miles, kg↔lbs, m↔ft, L↔gal, cm↔in
  - **Temperature**: °C↔°F↔K (all directions)
- **Design**: Cascading logic — checks temperature first, then unit tables, then falls back to currency API

### 4. 🕒 World Clock — `get_current_time`
- **Backend**: Python's `zoneinfo` (stdlib, no external dependency)
- **Coverage**: 50+ pre-mapped cities (Tokyo, New York, Karachi, Dubai, etc.) + any valid IANA timezone string
- **Output**: Formatted date, time (12-hour), and timezone with UTC offset

### 5. 📄 PDF RAG Pipeline — `query_pdf_documents`
This is the most technically sophisticated tool. It implements a complete **Retrieval-Augmented Generation** pipeline:

#### RAG Pipeline Flow:
```
PDF Upload → PyPDFLoader → Text Extraction
    → RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
        → HuggingFace Embeddings (all-MiniLM-L6-v2)
            → ChromaDB Vector Store
                → Similarity Search (top-k=4)
                    → Context Injection into LLM Prompt
                        → Grounded Answer Generation
```

#### Key RAG Design Decisions:
| Parameter | Value | Rationale |
|---|---|---|
| Chunk Size | 1000 chars | Balances context richness with retrieval precision |
| Chunk Overlap | 150 chars | Prevents information loss at chunk boundaries |
| Top-K Retrieval | 4 documents | Enough context without overwhelming the LLM's context window |
| Embedding Model | `all-MiniLM-L6-v2` | Runs 100% locally — no API calls, no data leaving your machine |
| Vector Store | ChromaDB (in-memory) | Zero configuration, instant setup, perfect for prototyping |

#### Privacy-First Design:
- All embeddings are computed **locally** using HuggingFace models
- PDF content **never leaves your machine** for embedding generation
- Only the retrieved context chunks are sent to Groq for answer generation
- Documents are cached via `@st.cache_resource` for performance

---

## 📂 Project Structure

```
conversational-bot/
├── conversaional-agent-app.py   # Main application (373 lines — single-file architecture)
├── requirements.txt             # Python dependencies
├── .env                         # API keys (git-ignored)
├── .env.example                 # Template for environment variables
├── .gitignore                   # Security-aware git ignore rules
├── README.md                    # Project documentation
├── docs/                        # PDF upload directory (git-ignored)
│   └── *.pdf                    # User-uploaded documents
└── .venv/                       # Python virtual environment
```

---

## ⚙️ Environment Configuration

The application is fully configurable via environment variables:

```env
GROQ_API_KEY=gsk_...           # Your Groq API key (required)
GROQ_MODEL=llama-3.1-70b-versatile  # LLM model selection
PDF_DOCS_DIRECTORY=./docs      # Where PDFs are stored
PDF_CHUNK_SIZE=1000            # RAG chunk size in characters
PDF_CHUNK_OVERLAP=150          # Overlap between chunks
PDF_RETRIEVAL_K=4              # Number of chunks to retrieve
```

---

## 🚀 Key Technical Highlights

### 1. Single-File Architecture
The entire application is contained in **one 373-line Python file**. This was a deliberate design choice — it makes the project easy to understand, deploy, and share while still being production-capable.

### 2. Groq's Speed Advantage
Groq's custom **LPU (Language Processing Unit)** hardware delivers inference speeds of ~300 tokens/second — roughly 10x faster than traditional GPU-based inference. This makes the agent feel **instantaneous** to the user.

### 3. Inline PDF Upload (No Sidebar)
Unlike most Streamlit AI apps that use sidebar file uploaders, this project uses Streamlit's `accept_file` parameter on `st.chat_input()` to allow **inline PDF attachment** directly in the chat bar — a much more natural UX.

### 4. Lazy Loading for Performance
HuggingFace embedding models (~80MB) are imported lazily — only when the RAG tool is actually needed. This keeps startup time fast for users who don't need the PDF feature.

### 5. Cache-Aware Vector Store
The ChromaDB vector store is wrapped in `@st.cache_resource`, so PDFs are only processed once. When new PDFs are uploaded, the cache is intelligently cleared and rebuilt.

### 6. Production-Ready Security
- `.env` file for secrets (never committed to git)
- `.gitignore` properly configured for Python, Streamlit, and data directories
- User documents stored in a git-ignored `docs/` directory

---

## 🔮 Future Roadmap

- **Persistent Vector Store**: Move from in-memory ChromaDB to a persistent database for cross-session document memory
- **Multi-format Document Support**: Extend RAG to support DOCX, TXT, CSV, and web URLs
- **Streaming Responses**: Implement token-by-token streaming for an even more responsive UX
- **Authentication**: Add user login for multi-tenant document isolation
- **Deployment**: Containerize with Docker and deploy to cloud (AWS/GCP/Azure)
- **Advanced RAG**: Implement hybrid search (semantic + keyword), re-ranking, and query decomposition

---

## 💡 Key Takeaways

1. **Agentic AI is the future** — LLMs that can reason about and use tools are fundamentally more powerful than static chatbots
2. **Groq makes AI fast** — LPU-based inference removes the latency bottleneck that makes AI feel sluggish
3. **RAG keeps AI grounded** — by retrieving from your actual documents, the agent gives accurate, source-backed answers instead of hallucinating
4. **LangGraph simplifies agent development** — the ReAct pattern gives you a production-grade agent with minimal boilerplate
5. **You don't need expensive APIs for embeddings** — HuggingFace's open-source models run locally and produce excellent results

---

## 🛠️ Built With

`Python` · `Streamlit` · `LangGraph` · `LangChain` · `Groq` · `Llama 3.1 70B` · `ChromaDB` · `HuggingFace Embeddings` · `Pydantic` · `Open-Meteo API` · `ExchangeRate API` · `Wikipedia API`

---

*This project was built as a hands-on exploration of agentic AI, RAG pipelines, and high-speed LLM inference — demonstrating how modern AI tooling can be composed into a genuinely useful, production-capable application.*
