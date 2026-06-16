import time
import json
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END
from config import AgentState, generator_llm, evaluator_llm
from engine import builder
from agents import get_message_text


def response_generator_node(state: AgentState):
    """Synthesizes a polished final response from all collected agent data."""
    t0 = time.time()
    original_user_query = get_message_text(state["messages"][0].content)

    # Collect all sub-agent evidence messages
    evidence_parts = []
    for msg in state["messages"]:
        msg_text = get_message_text(msg.content)
        if "Data Output" in msg_text:
            evidence_parts.append(msg_text)

    grounding_evidence = "\n\n".join(evidence_parts)

    if not grounding_evidence.strip():
        prompt = f"Answer directly, clearly and professionally:\n\nUser Query: {original_user_query}"
    else:
        prompt = f"""
        You are the Core Response Synthesizer Agent. Your job is to construct a beautifully polished,
        highly accurate final response resolving the user's primary objective using the
        collected agent telemetry logs below.

        User Primary Objective: {original_user_query}

        Collected Agent Evidence:
        \"\"\"
        {grounding_evidence}
        \"\"\"

        Instructions:
        - Synthesize all evidence into a clean, well-structured Markdown response.
        - Prioritise numerical accuracy and remove redundancies.
        - Use headers, bullet points and bold text where appropriate.
        - Do NOT mention internal agent names or pipeline terminology.
        """

    generation = generator_llm.invoke([HumanMessage(content=prompt)])
    duration_ms = round((time.time() - t0) * 1000, 2)

    logs = list(state.get("execution_logs", []))
    logs.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "agent": "Response Generator",
        "status": "success",
        "details": "Final response synthesized from all agent evidence.",
        "duration_ms": duration_ms
    })

    return {
        "final_response": get_message_text(generation.content),
        "execution_logs": logs,
        "current_task": "Running evaluation benchmarks..."
    }


def evaluation_node(state: AgentState):
    """Quality Auditor: scores the final response for accuracy and hallucination."""
    t0 = time.time()
    original_user_query = get_message_text(state["messages"][0].content)

    evidence_parts = [
        get_message_text(msg.content)
        for msg in state["messages"]
        if "Data Output" in get_message_text(msg.content)
    ]
    grounding_evidence = "\n".join(evidence_parts)
    final_output = state.get("final_response", "")

    judge_prompt = f"""
    You are the Lead Quality Auditor and Fact-Checking Supervisor.
    Analyze the pipeline artifacts and score the final output.

    1. Original User Task: {original_user_query}
    2. Grounding Evidence Logs: {grounding_evidence}
    3. Final System Output: {final_output}

    Respond STRICTLY in valid JSON — no markdown fences, no preamble:

    {{
        "accuracy": <integer 0-100>,
        "hallucination_rate": <integer 0-100>,
        "completeness": <integer 0-100>,
        "reasoning": "<one sentence justification>"
    }}
    """

    try:
        raw_reply = evaluator_llm.invoke([HumanMessage(content=judge_prompt)]).content
        reply_str = get_message_text(raw_reply).strip()

        # Strip markdown code fences if present
        if reply_str.startswith("```json"):
            reply_str = reply_str[7:].strip()
        if reply_str.startswith("```"):
            reply_str = reply_str[3:].strip()
        if reply_str.endswith("```"):
            reply_str = reply_str[:-3].strip()

        metrics = json.loads(reply_str)
    except Exception as e:
        metrics = {
            "accuracy": "N/A",
            "hallucination_rate": "N/A",
            "completeness": "N/A",
            "reasoning": f"Evaluation parse error: {str(e)}"
        }

    duration_ms = round((time.time() - t0) * 1000, 2)
    logs = list(state.get("execution_logs", []))
    logs.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "agent": "Evaluation Agent",
        "status": "success",
        "details": (
            f"Accuracy={metrics.get('accuracy')}, "
            f"Hallucination={metrics.get('hallucination_rate')}, "
            f"Completeness={metrics.get('completeness')}"
        ),
        "duration_ms": duration_ms
    })

    tracking_str = (
        f"Accuracy: {metrics.get('accuracy')}/100 | "
        f"Hallucination Rate: {metrics.get('hallucination_rate')}/100 | "
        f"Completeness: {metrics.get('completeness')}/100"
    )

    return {
        "messages": [AIMessage(content=tracking_str)],
        "evaluation_metrics": metrics,
        "execution_logs": logs,
        "current_task": "Finished"
    }


# ── Wire final nodes into the graph ───────────────────────────────────────────
builder.add_node("response_generator", response_generator_node)
builder.add_node("evaluator", evaluation_node)
builder.add_edge("response_generator", "evaluator")
builder.add_edge("evaluator", END)

compiled_graph = builder.compile()