import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# LangChain & LangGraph imports
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages

# Setup tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv papers")
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
tavily = TavilySearchResults()
tools = [arxiv, wiki, tavily]

# Setup LLM
llm = ChatGroq(model="qwen-qwq-32b")
llm_with_tools = llm.bind_tools(tools=tools)

# State schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node definition
def ai_assistance(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("AI_Assistance", ai_assistance)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "AI_Assistance")
builder.add_conditional_edges("AI_Assistance", tools_condition)
builder.add_edge("tools", END)
graph = builder.compile()

# Streamlit UI
st.set_page_config(page_title="Multi-Tool AI Chatbot", layout="wide")
st.title(" Multi-Tool AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_ai" not in st.session_state:
    st.session_state.pending_ai = None

# User input
user_input = st.text_input("Ask me:", "")

# Handle user submission
if st.button("Search") and user_input and not st.session_state.pending_ai:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history]
    result = graph.invoke({"messages": messages})
    ai_message = result["messages"][-1].content if hasattr(result["messages"][-1], "content") else str(result["messages"][-1])
    st.session_state.pending_ai = ai_message

# Human-in-the-loop review
if st.session_state.pending_ai:
    st.markdown("**AI Suggestion (edit or approve):**")
    edited_ai = st.text_area("AI's response:", st.session_state.pending_ai)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve & Add to Chat"):
            st.session_state.chat_history.append({"role": "assistant", "content": edited_ai})
            st.session_state.pending_ai = None
            st.rerun()  # <-- updated here
    with col2:
        if st.button("Reject & Retry"):
            st.session_state.pending_ai = None
            st.rerun()  # <-- updated here

# Display chat history
# st.markdown("---")
# for msg in reversed(st.session_state.chat_history):
#     if msg["role"] == "user":
#         st.markdown(f"** You:** {msg['content']}")
#     else:
#         st.markdown(f"** AI:** {msg['content']}")

st.markdown("---")
history = st.session_state.chat_history
pairs = []

# Group messages into (user, ai) pairs
i = 0
while i < len(history):
    if history[i]["role"] == "user":
        user_msg = history[i]["content"]
        ai_msg = history[i+1]["content"] if i+1 < len(history) and history[i+1]["role"] == "assistant" else ""
        pairs.append((user_msg, ai_msg))
        i += 2
    else:
        i += 1

for user_msg, ai_msg in reversed(pairs):
    st.markdown(f"** You:** {user_msg}")
    if ai_msg:
        st.markdown(f"** AI:** {ai_msg}")
