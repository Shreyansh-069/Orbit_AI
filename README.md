# 🔭 Orbit AI — Your Research Co-Pilot That Actually Knows Who To Ask

> Ask a question. Drop a PDF. Throw in an image. Orbit figures out which specialist agent should handle what — then hands you one clean, synthesized, fact-checked answer instead of three messy ones.

No more juggling separate tools for "summarize this PDF," "what's in this image," and "search this for me." One query box. One pipeline. One answer — built by a supervisor that knows exactly which agent to wake up for the job.

### 🌐 [Try it live → orbit-ai.streamlit.app](https://orbit-ai.streamlit.app/)

---

## ✨ What Makes Orbit Different

Most "AI assistant" demos either hardcode a single tool or blindly call every tool every time, wasting calls and money. Orbit does neither — it **routes**.

| | |
|---|---|
| 🧠 | **Smart routing** — a planner inspects your inputs and only activates the agents actually needed |
| 📄 | **PDF Agent** — extracts and reasons over up to 15 pages of any uploaded document |
| 🖼 | **Vision Agent** — reads charts, screenshots, photos, and scanned text using Gemini Vision |
| 🌐 | **Web Agent** — pulls real-time grounding from the live web via Tavily, every single run |
| ✨ | **Synthesizer** — merges every agent's findings into one polished, de-duplicated Markdown answer |
| 🎯 | **Evaluator** — independently scores the final answer for accuracy, hallucination risk, and completeness |
| 📊 | **Live pipeline map** — watch exactly which agent is active, done, or skipped, in real time |
| 🪵 | **Full execution logs** — timestamps, durations, and a visual timeline bar for every run |

---

## 🖥 See It In Action

1. Type a question (or just attach a file and ask "what's in this?")
2. Watch the **live agent flow map** light up as the Supervisor routes your request
3. Get a synthesized answer in the **Response** tab, complete with cited web sources
4. Dig into **Metrics**, **Execution Logs**, or raw **Logs** (JSON) if you want the receipts

---

## 🏗 How It's Built

Orbit is a **LangGraph state machine** wrapped in a **Streamlit** interface:

```
                    ┌──────────────┐
   User Query  ──▶  │  Supervisor  │  (planner_node)
   + PDF/Image       └──────┬───────┘
                            │ inspects inputs, builds task queue
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        📄 PDF Agent   🖼 Vision Agent  🌐 Web Agent
       (RAG-style       (Gemini Vision)  (Tavily +
        extraction)                       synthesis)
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                    ✨ Synthesizer
                  (merges all evidence
                   into one clean answer)
                            │
                            ▼
                    🎯 Evaluator
              (scores accuracy, hallucination,
                     completeness)
                            │
                            ▼
                    Final Response
                  → rendered in Streamlit
```

Each specialist agent only runs if it's actually relevant — no PDF uploaded means the PDF Agent is skipped entirely (and shown as "skipped," not just absent, in the flow map). The Web Agent always runs to keep every answer grounded in current information.

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful, conditional multi-agent graph |
| LLM | Google **Gemini** (`gemini-3.1-flash-lite`) via `langchain-google-genai` |
| Web Search | [Tavily](https://tavily.com) real-time search API |
| PDF Parsing | `pypdf` |
| UI | [Streamlit](https://streamlit.io) |
| Image Handling | `Pillow` |

---

## 🚀 Getting It Running

### 1. Clone & install

Orbit uses [`uv`](https://docs.astral.sh/uv/) for dependency management — fast, reproducible, no `pip` ceremony.

```bash
git clone https://github.com/your-username/orbit-ai.git
cd orbit-ai
uv sync
```

Don't have `uv` yet?

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

### 2. Add your API keys

Orbit needs two free API keys to function:

| Key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) |

**Pick whichever setup fits where you're running it:**

**🖥 Running locally** — create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

**☁️ Deploying on Streamlit Community Cloud** — there's no `.env` file in the cloud, so set your keys in **Settings → Secrets** in the Streamlit dashboard instead:
```toml
GOOGLE_API_KEY = "your_key_here"
TAVILY_API_KEY = "your_key_here"
```
Orbit reads keys straight from the environment, so it doesn't care whether they came from `.env`, Streamlit secrets, or were typed into the sidebar — all three work, and the sidebar will simply show "✓ Both API keys are set" once it finds them.

**🔑 No keys configured anywhere?** No problem — Orbit's sidebar has a built-in API key entry field, so anyone running the app can paste in their own keys for that session without touching any config file.

### 3. Run it

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` and you're live.

---

## 📁 Project Structure

```
orbit-ai/
├── app.py           # Streamlit UI — sidebar, live flow map, tabs, styling
├── config.py        # Shared state schema + lazy API key / model factories
├── engine.py        # LangGraph builder — supervisor routing logic
├── agents.py        # The specialist agents: PDF, Vision, Web Research
├── main.py          # Synthesizer + Evaluator nodes, final graph compile
├── pyproject.toml   # Project dependencies (managed by uv)
└── uv.lock          # Locked, reproducible dependency versions
```

| File | What it actually does |
|---|---|
| `config.py` | Defines `AgentState` (the shared data every node reads/writes) and lazily builds Gemini/Tavily clients — only created the moment they're needed, so missing keys never crash the app on startup |
| `engine.py` | The `planner_node` — looks at what was uploaded, decides which agents belong in the queue, and wires up the conditional routing edges |
| `agents.py` | Three independent node functions: `doc_analysis_node`, `vision_agent_node`, `web_research_node` — each one logs its own timing, status, and output |
| `main.py` | `response_generator_node` (the Synthesizer) and `evaluation_node` (the Evaluator/judge), plus the final `compiled_graph` everything runs through |
| `app.py` | Everything visual: the live pipeline map, status pills, response/metrics/logs tabs, and the API key sidebar |

---

## 🧪 What's Actually Happening Under the Hood

1. Your query (plus optional PDF/image) becomes the initial `AgentState`
2. The **Supervisor** builds a `pending_tasks` queue based on what you attached
3. Each queued agent runs, appends its findings to a shared evidence pool, and updates `used_agents` + `execution_logs`
4. The **Synthesizer** reads *all* collected evidence and writes one coherent Markdown response — no agent names, no pipeline jargon, just the answer
5. The **Evaluator** acts as an independent judge, scoring the final answer against the original query and the raw evidence — catching potential hallucinations before you see them
6. Streamlit streams every step live, so the flow map and status pill update node-by-node instead of going silent until the whole thing finishes

---

## 🛣 Ideas for Where This Could Go Next

- Loop the Evaluator's score back into the graph — low scores trigger an automatic retry
- Swap the PDF Agent's flat text extraction for proper chunked vector RAG for longer documents
- Add a persistent vector-based memory layer for multi-turn follow-up questions
- Multi-document support (multiple PDFs in one run, not just one)

---

## 📄 License

MIT — do whatever you want with it, just don't blame me if the Evaluator gives your homework a 40/100.
