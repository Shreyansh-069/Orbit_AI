import os
import time
import json
import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage
from main import compiled_graph

st.set_page_config(
    page_title="Orbit AI · Research Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
header[data-testid="stHeader"] { background: transparent !important; }

button[data-testid="collapsedControl"] {
    background: #1e1e35 !important; color: #a78bfa !important;
    border: 1px solid #2e2e50 !important; border-radius: 6px !important;
}
button[data-testid="collapsedControl"]:hover { background: #2a2a45 !important; }

.stApp { background: #0c0c14; color: #e2e8f0; }

section[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e30; }
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa !important; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;
}
section[data-testid="stSidebar"] .stTextArea textarea {
    background: #111128 !important; border: 1px solid #1e1e35 !important;
    border-radius: 8px !important; color: #e2e8f0 !important; font-size: 0.87rem !important;
}
section[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: #7c3aed !important; box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.tip-box {
    background: #111128; border: 1px solid #1e1e35; border-radius: 8px;
    padding: 0.75rem 0.9rem; font-size: 0.76rem; color: #94a3b8 !important;
    line-height: 1.55; margin-bottom: 0.5rem;
}
.tip-box b { color: #a78bfa !important; }

.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px; }

.orbit-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1rem 1.4rem; background: #0f0f1a;
    border: 1px solid #1e1e30; border-radius: 14px;
    margin-bottom: 1.4rem;
}
.orbit-title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; }
.orbit-subtitle { font-size: 0.74rem; color: #64748b; margin-top: 2px; }

.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 9999px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}
.status-running { background: #1e293b; color: #38bdf8; border: 1px solid #0284c7; }
.status-success { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.status-error   { background: #2d0a0a; color: #f87171; border: 1px solid #dc2626; }

.response-card {
    background: linear-gradient(135deg, #0f0f1a 0%, #111128 100%);
    border: 1px solid #1e1e35; border-left: 4px solid #a78bfa;
    border-radius: 10px; padding: 1.8rem 2rem;
    font-size: 0.93rem; line-height: 1.75; color: #e2e8f0;
}

.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 1.4rem; }
.metric-card { background: #0f0f1a; border: 1px solid #1e1e30; border-radius: 10px; padding: 1rem 1.2rem; }
.metric-label { font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 6px; }
.metric-value { font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono',monospace; color: #f1f5f9; line-height: 1.1; }
.metric-unit  { font-size: 0.73rem; color: #475569; font-weight: 400; margin-left: 3px; }
.metric-accent-purple { border-top: 3px solid #a78bfa; }
.metric-accent-green  { border-top: 3px solid #4ade80; }
.metric-accent-red    { border-top: 3px solid #f87171; }
.metric-accent-blue   { border-top: 3px solid #38bdf8; }

/* ── Flow map ── */
.flow-map-wrap {
    background: #0f0f1a; border: 1px solid #1e1e30; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.flow-map-title {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: #475569; margin-bottom: 1.2rem;
}
.flow-outer-row {
    display: flex; align-items: center; justify-content: center;
    gap: 10px; flex-wrap: wrap;
}
.flow-node { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.flow-bubble {
    width: 68px; height: 68px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; border: 2.5px solid transparent;
}
.flow-bubble.idle    { background: #111128; border-color: #1e1e35; }
.flow-bubble.active  { background: #1e1b4b; border-color: #818cf8; box-shadow: 0 0 0 4px rgba(129,140,248,0.2); }
.flow-bubble.done    { background: #052e16; border-color: #22c55e; }
.flow-bubble.skipped { background: #111128; border-color: #1e1e35; opacity: 0.35; }
.flow-label { font-size: 0.65rem; font-weight: 600; color: #475569; text-align: center; text-transform: uppercase; letter-spacing: 0.06em; }
.flow-label.active  { color: #818cf8; }
.flow-label.done    { color: #4ade80; }
.flow-arrow { color: #2e2e50; font-size: 1.3rem; margin-top: -20px; }
.sub-agents-group { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.sub-agents-row { display: flex; align-items: center; gap: 10px; }
.sub-agents-label { font-size: 0.62rem; color: #334155; letter-spacing: 0.05em; text-transform: uppercase; }

.pipeline-route {
    display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
    padding: 0.9rem 1.1rem; background: #0f0f1a;
    border: 1px solid #1e1e30; border-radius: 10px;
}
.route-node {
    background: #1e1e35; color: #c4b5fd; padding: 4px 12px;
    border-radius: 6px; font-size: 0.75rem; font-weight: 500;
    font-family: 'JetBrains Mono',monospace; border: 1px solid #312e55;
}
.route-arrow { color: #475569; font-size: 0.85rem; }

.log-table { width: 100%; border-collapse: collapse; font-size: 0.79rem; font-family: 'JetBrains Mono',monospace; }
.log-table th {
    background: #111128; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.64rem;
    padding: 8px 12px; text-align: left; border-bottom: 1px solid #1e1e30;
}
.log-table td { padding: 10px 12px; border-bottom: 1px solid #131325; color: #cbd5e1; vertical-align: top; }
.log-table tr:last-child td { border-bottom: none; }
.log-table tr:hover td { background: #0f0f1e; }
.badge-success { color: #4ade80; font-weight: 700; }
.badge-error   { color: #f87171; font-weight: 700; }
.badge-skipped { color: #94a3b8; font-weight: 700; }

.source-chip {
    display: inline-block; background: #1e1e35; color: #93c5fd;
    border: 1px solid #1e3a5f; border-radius: 4px; padding: 2px 8px;
    font-size: 0.7rem; margin: 2px 2px; font-family: 'JetBrains Mono',monospace;
    word-break: break-all; text-decoration: none;
}
.source-chip:hover { background: #253449; }

.agent-report {
    background: #0a0a12; border: 1px solid #1e1e30; border-radius: 8px;
    padding: 1rem 1.2rem; font-size: 0.82rem; color: #94a3b8;
    line-height: 1.65; white-space: pre-wrap; font-family: 'Inter',sans-serif;
}

.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; border-bottom: 1px solid #1e1e30; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 6px 6px 0 0; font-size: 0.82rem; font-weight: 500; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: #1e1e35 !important; color: #c4b5fd !important; border-bottom: 2px solid #a78bfa !important; }

.stButton > button {
    background: linear-gradient(135deg,#7c3aed,#5b21b6) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.6rem 1rem !important; transition: all 0.15s ease !important; letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#6d28d9,#4c1d95) !important;
    transform: translateY(-1px) !important;
}

hr { border-color: #1e1e30 !important; }
.stInfo, .stWarning, .stSuccess, .stError { border-radius: 8px !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Flow map builder ───────────────────────────────────────────────────────────
def node_state(name, used, active):
    if name == active:
        return "active"
    if name in used:
        return "done"
    return "idle"

def make_bubble(icon, label, state):
    return (
        '<div class="flow-node">'
        '<div class="flow-bubble ' + state + '">' + icon + '</div>'
        '<div class="flow-label ' + state + '">' + label + '</div>'
        '</div>'
    )

def build_flow_map(used_agents=None, active_node=None):
    used = set(used_agents or [])

    sup   = make_bubble("🧠", "Supervisor", node_state("supervisor",   used, active_node))
    pdf   = make_bubble("📄", "PDF",        node_state("pdf_agent",    used, active_node))
    img   = make_bubble("🖼", "Image",      node_state("image_agent",  used, active_node))
    web   = make_bubble("🌐", "Web",        node_state("web_agent",    used, active_node))
    synth = make_bubble("✨", "Synthesizer",node_state("synthesizer",  used, active_node))
    evl   = make_bubble("🎯", "Evaluator",  node_state("evaluator",    used, active_node))

    html = (
        '<div class="flow-map-wrap">'
        '<div class="flow-map-title">Agent Pipeline &middot; Live View</div>'
        '<div class="flow-outer-row">'
        + sup +
        '<div class="flow-arrow">&#8594;</div>'
        '<div class="sub-agents-group">'
        '<div class="sub-agents-row">'
        + pdf + img + web +
        '</div>'
        '<div class="sub-agents-label">Specialist Agents</div>'
        '</div>'
        '<div class="flow-arrow">&#8594;</div>'
        + synth +
        '<div class="flow-arrow">&#8594;</div>'
        + evl +
        '</div>'
        '</div>'
    )
    return html


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="orbit-header">'
    '<div style="font-size:2rem">🔭</div>'
    '<div>'
    '<div class="orbit-title">Orbit AI &middot; Research Assistant</div>'
    '<div class="orbit-subtitle">Multi-agent &middot; Web search &middot; PDF analysis &middot; Image understanding</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Your Query")
    user_query = st.text_area(
        label="Query",
        placeholder="Ask anything — e.g. 'Summarise this PDF', 'What's in this image?', 'Latest news on AI...'",
        height=130,
        label_visibility="collapsed"
    )

    st.markdown("### 📎 Attachments *(optional)*")
    uploaded_pdf   = st.file_uploader("Upload a PDF document", type=["pdf"], label_visibility="visible")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"], label_visibility="visible")

    st.markdown("---")

    st.markdown("### 💡 What can I do?")
    st.markdown(
        '<div class="tip-box"><b>📝 Ask a question directly</b><br>Type any question and the web agent will search for a real-time answer.</div>'
        '<div class="tip-box"><b>📄 Analyse a PDF</b><br>Upload a PDF and ask things like:<br><i>"Summarise this document"</i> or <i>"What are the key findings?"</i></div>'
        '<div class="tip-box"><b>🖼 Understand an image</b><br>Upload a photo or screenshot and ask:<br><i>"What is shown here?"</i> or <i>"Describe the chart in this image"</i></div>'
        '<div class="tip-box"><b>🔀 Combine sources</b><br>Upload a PDF and an image together, then ask a question that spans both.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    active = []
    if uploaded_pdf:   active.append("📄 " + uploaded_pdf.name)
    if uploaded_image: active.append("🖼 " + uploaded_image.name)
    if active:
        st.markdown("**Loaded files:**")
        for a in active:
            st.markdown('<div style="font-size:0.78rem;color:#94a3b8;padding:2px 0">' + a + '</div>', unsafe_allow_html=True)
        st.markdown("")

    submit_button = st.button("▶  Run Pipeline", use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────
if submit_button:
    if not user_query.strip():
        st.warning("⚠️ Please enter a query before running the pipeline.")
        st.stop()

    temp_img_path, temp_pdf_path = None, None

    if uploaded_image:
        temp_img_path = "temp_" + uploaded_image.name
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

    if uploaded_pdf:
        temp_pdf_path = "temp_" + uploaded_pdf.name
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

    if uploaded_image or uploaded_pdf:
        col_img, col_pdf = st.columns([1, 1])
        if uploaded_image:
            with col_img:
                st.image(Image.open(temp_img_path), caption="Active image", width=260)
        if uploaded_pdf:
            with col_pdf:
                st.markdown(
                    '<div style="background:#0f0f1a;border:1px solid #1e1e30;border-radius:8px;'
                    'padding:1rem 1.2rem;font-size:0.83rem;color:#94a3b8;margin-top:8px">'
                    '📄 <strong style="color:#e2e8f0">' + uploaded_pdf.name + '</strong><br>'
                    '<span style="font-size:0.72rem">Ready for PDF analysis</span></div>',
                    unsafe_allow_html=True
                )
        st.markdown("")

    status_placeholder  = st.empty()
    flowmap_placeholder = st.empty()

    status_placeholder.markdown(
        '<div class="status-pill status-running">&#x21BB; &nbsp;Initialising pipeline&hellip;</div>',
        unsafe_allow_html=True
    )
    flowmap_placeholder.markdown(build_flow_map(), unsafe_allow_html=True)

    initial_state = {
        "messages":       [HumanMessage(content=user_query)],
        "image_path":     temp_img_path or "",
        "pdf_path":       temp_pdf_path or "",
        "current_task":   "Initialising...",
        "used_agents":    [],
        "agent_reports":  {},
        "pending_tasks":  [],
        "execution_logs": [],
        "start_time":     time.time()
    }

    pipeline_start = time.time()

    try:
        final_state = {**initial_state}

        for output in compiled_graph.stream(initial_state):
            for node_name, updated_values in output.items():
                final_state = {**final_state, **updated_values}
                task_label  = final_state.get("current_task", "Processing...")
                used_agents = final_state.get("used_agents", [])

                if task_label not in ("Finished", "Idle"):
                    status_placeholder.markdown(
                        '<div class="status-pill status-running">&#x21BB; &nbsp;'
                        + node_name.upper() + ' &mdash; ' + task_label + '</div>',
                        unsafe_allow_html=True
                    )
                    flowmap_placeholder.markdown(
                        build_flow_map(used_agents=used_agents, active_node=node_name),
                        unsafe_allow_html=True
                    )

        total_time  = time.time() - pipeline_start
        used_agents = final_state.get("used_agents", [])

        status_placeholder.markdown(
            '<div class="status-pill status-success">&#10003; &nbsp;Pipeline complete</div>',
            unsafe_allow_html=True
        )
        flowmap_placeholder.markdown(
            build_flow_map(used_agents=used_agents, active_node=None),
            unsafe_allow_html=True
        )

        metrics = final_state.get("evaluation_metrics", {})

        tab_response, tab_metrics, tab_logs, tab_json = st.tabs([
            "💬  Response",
            "📊  Metrics",
            "🪵  Execution Logs",
            "{ }  Logs"
        ])

        # TAB 1 — RESPONSE
        with tab_response:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            final_resp = final_state.get("final_response", "_No response generated._")
            st.markdown('<div class="response-card">' + final_resp + '</div>', unsafe_allow_html=True)

            sources = final_state.get("agent_reports", {}).get("web_agent_sources", [])
            if sources:
                st.markdown(
                    '<div style="margin-top:1.2rem;font-size:0.72rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:0.08em">Web sources</div>',
                    unsafe_allow_html=True
                )
                chips = "".join(
                    '<a href="' + s + '" target="_blank" class="source-chip">'
                    + (s[:60] + "…" if len(s) > 60 else s) + '</a>'
                    for s in sources
                )
                st.markdown(chips, unsafe_allow_html=True)

        # TAB 2 — METRICS
        with tab_metrics:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

            acc  = metrics.get("accuracy",           "N/A")
            hal  = metrics.get("hallucination_rate", "N/A")
            comp = metrics.get("completeness",       "N/A")

            st.markdown(
                '<div class="metric-grid">'
                '<div class="metric-card metric-accent-green">'
                '<div class="metric-label">Accuracy</div>'
                '<div class="metric-value">' + str(acc) + '<span class="metric-unit">/100</span></div></div>'
                '<div class="metric-card metric-accent-red">'
                '<div class="metric-label">Hallucination Rate</div>'
                '<div class="metric-value">' + str(hal) + '<span class="metric-unit">/100</span></div></div>'
                '<div class="metric-card metric-accent-purple">'
                '<div class="metric-label">Completeness</div>'
                '<div class="metric-value">' + str(comp) + '<span class="metric-unit">/100</span></div></div>'
                '<div class="metric-card metric-accent-blue">'
                '<div class="metric-label">Total Latency</div>'
                '<div class="metric-value">' + str(round(total_time, 1)) + '<span class="metric-unit">s</span></div></div>'
                '</div>',
                unsafe_allow_html=True
            )

            reasoning = metrics.get("reasoning", "")
            if reasoning:
                st.markdown(
                    '<div style="font-size:0.8rem;color:#64748b;padding:0.8rem 1rem;'
                    'background:#0f0f1a;border:1px solid #1e1e30;border-radius:8px">'
                    '<strong style="color:#94a3b8">Evaluator reasoning:</strong> ' + reasoning + '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div style="margin:1.2rem 0 0.5rem;font-size:0.72rem;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.08em">Pipeline route</div>',
                unsafe_allow_html=True
            )

            route_html = '<div class="pipeline-route">'
            for i, agent in enumerate(used_agents):
                route_html += '<span class="route-node">' + agent + '</span>'
                if i < len(used_agents) - 1:
                    route_html += '<span class="route-arrow">&#8594;</span>'
            route_html += '</div>'
            st.markdown(route_html, unsafe_allow_html=True)

            reports      = final_state.get("agent_reports", {})
            display_keys = {k: v for k, v in reports.items() if not k.endswith("_sources")}
            if display_keys:
                st.markdown(
                    '<div style="margin:1.2rem 0 0.5rem;font-size:0.72rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:0.08em">Sub-agent summaries</div>',
                    unsafe_allow_html=True
                )
                for key, content in display_keys.items():
                    with st.expander("⚙  " + key.upper().replace("_", " "), expanded=False):
                        st.markdown('<div class="agent-report">' + content + '</div>', unsafe_allow_html=True)

        # TAB 3 — EXECUTION LOGS
        with tab_logs:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            logs = final_state.get("execution_logs", [])

            if logs:
                rows = ""
                for entry in logs:
                    status      = entry.get("status", "unknown")
                    badge_class = {"success": "badge-success", "error": "badge-error", "skipped": "badge-skipped"}.get(status, "")
                    status_icon = {"success": "✓", "error": "✗", "skipped": "⊘"}.get(status, "·")
                    rows += (
                        "<tr>"
                        "<td style='color:#475569'>" + str(entry.get("timestamp", "—")) + "</td>"
                        "<td><strong style='color:#c4b5fd'>" + str(entry.get("agent", "—")) + "</strong></td>"
                        "<td class='" + badge_class + "'>" + status_icon + " " + status.upper() + "</td>"
                        "<td style='color:#94a3b8;font-size:0.76rem'>" + str(entry.get("details", "—")) + "</td>"
                        "<td style='color:#475569;text-align:right'>" + str(entry.get("duration_ms", "—")) + " ms</td>"
                        "</tr>"
                    )

                st.markdown(
                    '<table class="log-table"><thead><tr>'
                    '<th>Time</th><th>Agent</th><th>Status</th><th>Details</th>'
                    '<th style="text-align:right">Duration</th>'
                    '</tr></thead><tbody>' + rows + '</tbody></table>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div style="margin-top:1.5rem;font-size:0.72rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">'
                    'Execution timeline</div>',
                    unsafe_allow_html=True
                )
                total_ms = sum(e.get("duration_ms", 0) for e in logs if isinstance(e.get("duration_ms"), (int, float)))
                timeline_html = '<div style="display:flex;gap:3px;height:28px;border-radius:6px;overflow:hidden">'
                colors = ["#7c3aed","#0ea5e9","#10b981","#f59e0b","#ef4444"]
                for i, entry in enumerate(logs):
                    dur = entry.get("duration_ms", 0)
                    if not isinstance(dur, (int, float)) or total_ms == 0:
                        continue
                    pct        = max((dur / total_ms) * 100, 3)
                    color      = colors[i % len(colors)]
                    agent_name = str(entry.get("agent", ""))
                    label      = agent_name[:12]
                    timeline_html += (
                        '<div style="background:' + color + ';width:' + str(pct) + '%;display:flex;align-items:center;'
                        'justify-content:center;font-size:0.6rem;color:white;font-family:JetBrains Mono,monospace;'
                        'padding:0 4px;overflow:hidden;white-space:nowrap" title="' + agent_name + ' ' + str(dur) + 'ms">'
                        + label + '</div>'
                    )
                timeline_html += "</div>"
                st.markdown(timeline_html, unsafe_allow_html=True)
            else:
                st.info("No structured logs were generated during this run.")

        # TAB 4 — LOGS (JSON)
        with tab_json:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

            st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Execution logs</div>', unsafe_allow_html=True)
            st.json(final_state.get("execution_logs", []))

            st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem">Evaluation metrics</div>', unsafe_allow_html=True)
            st.json(final_state.get("evaluation_metrics", {}))

            st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem">Agent reports</div>', unsafe_allow_html=True)
            display_reports = {k: v for k, v in final_state.get("agent_reports", {}).items() if not k.endswith("_sources")}
            st.json(display_reports)

            st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem">Full pipeline state</div>', unsafe_allow_html=True)
            safe_state = {k: v for k, v in final_state.items() if k not in ("messages",) and isinstance(v, (str,int,float,bool,list,dict,type(None)))}
            with st.expander("Expand full state dump"):
                st.json(safe_state)

    except Exception as e:
        status_placeholder.markdown(
            '<div class="status-pill status-error">&#10007; &nbsp;Pipeline failed</div>',
            unsafe_allow_html=True
        )
        st.error("**Pipeline error:** " + str(e))

    finally:
        for path in [temp_img_path, temp_pdf_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

else:
    st.markdown(build_flow_map(), unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:3rem 2rem 4rem">'
        '<div style="font-size:0.95rem;color:#334155;font-weight:500">'
        'Enter a query in the sidebar and click <strong style="color:#7c3aed">Run Pipeline</strong></div>'
        '<div style="font-size:0.82rem;color:#1e293b;margin-top:0.5rem">'
        'Optionally attach a PDF or image to activate specialist agents</div>'
        '</div>',
        unsafe_allow_html=True
    )