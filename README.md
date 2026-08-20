# AI Conversational Agent with Streamlit and Groq

A robust, multi-tool conversational AI assistant built with **LangGraph**, **Streamlit**, and **Groq**. 

The agent is capable of making intelligent decisions about when to look up world knowledge, check live exchange rates, fetch real-time weather, query world time zones, or search through your locally uploaded PDF documents.

## Features & Tools

1. **🌤️ Live Weather (`get_current_temperature`)**: Fetches real-time temperature data using the free Open-Meteo API.
2. **📚 Wikipedia Search (`search_wikipedia`)**: Queries Wikipedia for comprehensive world knowledge.
3. **💱 Currency & Unit Converter (`convert_currency_or_unit`)**: Provides live exchange rate conversions (via exchangerate-api) and standard unit conversions (km to miles, Celsius to Fahrenheit, etc.).
4. **🕒 World Clock (`get_current_time`)**: Uses Python's native `zoneinfo` to fetch the current time for any city worldwide.
5. **📄 PDF Knowledge Base (`query_pdf_documents`)**: A fully local RAG (Retrieval-Augmented Generation) pipeline. Upload a PDF directly in the chat, and the app uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) and ChromaDB to let you ask questions about your documents.

## Tech Stack
- **Frontend**: Streamlit
- **LLM Engine**: Groq (Llama 3.1 70B)
- **Agent Orchestration**: LangGraph (`create_react_agent`)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`) via `langchain-huggingface`
- **Vector Store**: Chroma

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/conversational-bot.git
cd conversational-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Copy the `.env.example` file to `.env` and add your [Groq API Key](https://console.groq.com/keys).
```bash
cp .env.example .env
```

## Running the Application

Start the Streamlit dashboard:
```bash
streamlit run conversaional-agent-app.py
```
*(Note: You can rename `conversaional-agent-app.py` to `app.py` if preferred.)*

The app will open automatically in your browser. You can use the chat interface to ask questions, or use the **inline attachment (paperclip) icon** in the chat bar to upload PDF documents on the fly.
