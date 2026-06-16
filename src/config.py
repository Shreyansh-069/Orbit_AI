import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    raise ValueError(
        "Missing critical API keys. Ensure GOOGLE_API_KEY and TAVILY_API_KEY exist in your .env file."
    )


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    image_path: str
    pdf_path: str

    start_time: float
    current_task: str

    used_agents: list[str]
    agent_reports: dict

    pending_tasks: list[str]

    final_response: str
    evaluation_metrics: dict
    execution_logs: list[dict]


# Gemini model instances — using gemini-3.1-flash-lite as gemini-3.1-flash-lite is not a valid model ID
planner_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash -lite",
    temperature=0.0
)

generator_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.1
)

evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0 
)

tavily_search_tool = TavilySearchResults(
    max_results=4,
    topic="general"
)