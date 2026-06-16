import os
import time
import base64
from pypdf import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import AgentState, tavily_search_tool


def get_message_text(content) -> str:
    """Safely extracts a flat string from any LangChain message content structure."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def _append_log(state: AgentState, agent: str, status: str, details: str, duration_ms: float = 0) -> list[dict]:
    """Helper to build a new structured log entry."""
    logs = list(state.get("execution_logs", []))
    logs.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "agent": agent,
        "status": status,
        "details": details,
        "duration_ms": round(duration_ms, 2)
    })
    return logs


def doc_analysis_node(state: AgentState):
    """PDF Document Analysis Agent using text extraction + LLM summarization."""
    t0 = time.time()
    agent_label = "Document Agent (PDF RAG)"

    used_agents = list(state.get("used_agents", []))
    used_agents.append(agent_label)

    pdf_path = state.get("pdf_path")
    user_query = get_message_text(state["messages"][0].content)
    reports = dict(state.get("agent_reports", {}))
    pending = list(state.get("pending_tasks", []))

    if "doc_agent" in pending:
        pending.remove("doc_agent")

    if not pdf_path or not os.path.exists(pdf_path):
        logs = _append_log(state, agent_label, "skipped", "No PDF file provided or path invalid.", (time.time() - t0) * 1000)
        reports["doc_agent"] = "PDF analysis bypassed: No physical file provided."
        return {
            "used_agents": used_agents,
            "current_task": "Idle",
            "agent_reports": reports,
            "pending_tasks": pending,
            "execution_logs": logs
        }

    try:
        reader = PdfReader(pdf_path)
        raw_text = ""
        pages_read = min(len(reader.pages), 15)
        for i, page in enumerate(reader.pages[:pages_read]):
            text = page.extract_text()
            if text:
                raw_text += f"\n--- Page {i+1} ---\n{text}"

        if not raw_text.strip():
            raise ValueError("The uploaded PDF does not contain indexable/extractable text.")

        extractor = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
        prompt = f"""
        Analyze the raw PDF text document below and locate sections, values, names, or parameters 
        specifically relevant to satisfying the User Query.

        User Query: {user_query}

        Raw Document Text:
        {raw_text}
        """
        response = extractor.invoke([HumanMessage(content=prompt)])
        summary = get_message_text(response.content)

        output_msg = f"[Document Agent Data Output]: Context successfully extracted.\n{summary}"
        reports["doc_agent"] = summary
        logs = _append_log(
            state, agent_label, "success",
            f"Extracted text from {pages_read} page(s). LLM summarization complete.",
            (time.time() - t0) * 1000
        )

    except Exception as e:
        output_msg = f"[Document Agent Data Output]: PDF extraction error: {str(e)}"
        reports["doc_agent"] = f"Extraction failed: {str(e)}"
        logs = _append_log(state, agent_label, "error", str(e), (time.time() - t0) * 1000)

    return {
        "messages": [AIMessage(content=output_msg)],
        "used_agents": used_agents,
        "current_task": "Idle",
        "agent_reports": reports,
        "pending_tasks": pending,
        "execution_logs": logs
    }


def vision_agent_node(state: AgentState):
    """Multimodal Vision Agent using Gemini Vision for image analysis."""
    t0 = time.time()
    agent_label = "Vision Agent (Gemini Vision)"

    used_agents = list(state.get("used_agents", []))
    used_agents.append(agent_label)

    image_path = state.get("image_path")
    user_query = get_message_text(state["messages"][0].content)
    reports = dict(state.get("agent_reports", {}))
    pending = list(state.get("pending_tasks", []))

    if "vision_agent" in pending:
        pending.remove("vision_agent")

    if not image_path or not os.path.exists(image_path):
        logs = _append_log(state, agent_label, "skipped", "No image file provided or path invalid.", (time.time() - t0) * 1000)
        reports["vision_agent"] = "Vision bypassed: No image target provided."
        return {
            "used_agents": used_agents,
            "current_task": "Idle",
            "agent_reports": reports,
            "pending_tasks": pending,
            "execution_logs": logs
        }

    try:
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        # Detect image mime type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/jpeg")

        vision_model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
        prompt = f"""
        Perform a comprehensive visual scan. Locate and extract key figures, written text, symbols,
        objects, or structural data directly answering the User Query.

        User Query: {user_query}
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_data}"}
            ]
        )
        response = vision_model.invoke([message])
        analysis = get_message_text(response.content)

        output_msg = f"[Vision Agent Data Output]: OCR image scan completed.\n{analysis}"
        reports["vision_agent"] = analysis
        logs = _append_log(
            state, agent_label, "success",
            f"Image processed ({mime_type}). Visual analysis extracted successfully.",
            (time.time() - t0) * 1000
        )

    except Exception as e:
        output_msg = f"[Vision Agent Data Output]: Vision processing error: {str(e)}"
        reports["vision_agent"] = f"Visual analysis failed: {str(e)}"
        logs = _append_log(state, agent_label, "error", str(e), (time.time() - t0) * 1000)

    return {
        "messages": [AIMessage(content=output_msg)],
        "used_agents": used_agents,
        "current_task": "Idle",
        "agent_reports": reports,
        "pending_tasks": pending,
        "execution_logs": logs
    }


def web_research_node(state: AgentState):
    """Web Research Agent powered by Tavily search + LLM synthesis."""
    t0 = time.time()
    agent_label = "Web Research Agent (Tavily)"

    used_agents = list(state.get("used_agents", []))
    used_agents.append(agent_label)

    user_query = get_message_text(state["messages"][0].content)
    reports = dict(state.get("agent_reports", {}))
    pending = list(state.get("pending_tasks", []))

    if "web_agent" in pending:
        pending.remove("web_agent")

    try:
        # Search directly — skipping LLM query-rewrite saves one full round-trip
        search_results = tavily_search_tool.invoke({"query": user_query[:400]})
        sources = [res["url"] for res in search_results]
        context_list = [f"Source: {res['url']}\nContent: {res['content']}" for res in search_results]
        raw_web_data = "\n---\n".join(context_list)

        synthesis_model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
        synthesis_prompt = (
            f"You are a web research agent. Extract only the key facts from the sources below "
            f"that directly answer the user query. Be concise.\n\n"
            f"User query: {user_query}\n\nSources:\n{raw_web_data}"
        )
        response = synthesis_model.invoke([HumanMessage(content=synthesis_prompt)])
        summary = get_message_text(response.content)

        output_msg = f"[Web Agent Data Output]: Real-time search succeeded.\n{summary}"
        reports["web_agent"] = summary
        reports["web_agent_sources"] = sources
        logs = _append_log(
            state, agent_label, "success",
            f"{len(search_results)} sources retrieved and synthesized.",
            (time.time() - t0) * 1000
        )

    except Exception as e:
        output_msg = f"[Web Agent Data Output]: Web lookup failed: {str(e)}"
        reports["web_agent"] = f"Web lookup failed: {str(e)}"
        logs = _append_log(state, agent_label, "error", str(e), (time.time() - t0) * 1000)

    return {
        "messages": [AIMessage(content=output_msg)],
        "used_agents": used_agents,
        "current_task": "Idle",
        "agent_reports": reports,
        "pending_tasks": pending,
        "execution_logs": logs
    }