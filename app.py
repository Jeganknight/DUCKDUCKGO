import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Available models for user selection
model_options = {
    "LLaMA 3.3 70B Versatile (Meta)": "llama-3.3-70b-versatile",
    "LLaMA 3.1 8B Instant (Meta)": "llama-3.1-8b-instant",
    "Gemma 2 9B IT (Google)": "gemma2-9b-it",
    "Qwen-2.5 32B (Alibaba Cloud)": "qwen-2.5-32b"

}

# Streamlit UI
st.set_page_config(page_title="DuckDuckGo Search", layout="wide")
st.title("🔍 DuckDuckGo AI Search")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox("Select a model:", list(model_options.keys()))
    st.markdown("---")
    st.write("📝 **Tip:** Use clear and specific queries for better results!")

# Input field for user query
user_query = st.text_input(
    "🔎 Enter your search query:",
    placeholder="e.g., Best Italian restaurants in Chennai",
    help="Enter a keyword or question to search using DuckDuckGo"
)

if user_query:
    model_id = model_options[selected_model]
    agent = Agent(
        model=Groq(id=model_id, api_key=groq_api_key),
        description=user_query,  # Directly passing the query
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

    st.subheader("🔎 Search Results")
    with st.spinner("Fetching results... Please wait ⏳"):
        try:
            agent_response = agent.run(user_query)
            if agent_response and hasattr(agent_response, "content"):
                st.write(agent_response.content)  # Display results
            else:
                st.warning("No results found. Try refining your query.")
        except Exception as e:
            st.error("⚠️ An error occurred. Please wait a minute and try again.")
