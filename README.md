# 🔭 Orbit AI — Multi-Agent Research Assistant

A multi-agent AI pipeline built with **LangGraph**, **Gemini Vision**, **Tavily**, and **PDF RAG**, wrapped in a Streamlit UI. Ask questions, upload PDFs, and analyse images — all in one place.

---

## ✨ Features

- **🧠 Supervisor Agent** — Orchestrates the pipeline, decides which specialist agents to invoke based on the query and available inputs
- **📄 PDF Agent** — Extracts and reasons over uploaded PDF documents using RAG (Retrieval-Augmented Generation)
- **🖼 Image Agent** — Understands and describes uploaded images using Gemini Vision
- **🌐 Web Agent** — Searches the web in real time using Tavily to answer current-events or factual queries
- **✨ Synthesizer** — Combines outputs from all active agents into a single coherent response
- **🎯 Evaluator** — Scores the final response on accuracy, hallucination rate, and completeness
- **Live agent flow map** — Visual pipeline showing which agents are idle, active, or done in real time

---

## 🗂 Project Structure

```
├── app.py              # Streamlit UI — all frontend logic and agent flow map
├── main.py             # LangGraph graph definition and compiled_graph export
├── agents.py           # Individual agent implementations (supervisor, pdf, image, web, synthesizer, evaluator)
├── config.py           # API keys, model names, and shared configuration
├── engine.py           # Pipeline execution helpers and state utilities
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shreyansh-069/orbit-ai.git
cd orbit-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Or set them directly in `config.py`.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🚀 Usage

| Goal | What to do |
|---|---|
| Ask a general question | Type your query and click **Run Pipeline** — the web agent searches in real time |
| Analyse a PDF | Upload a `.pdf` file, type your question, then run |
| Understand an image | Upload a `.jpg`, `.png`, or `.webp`, type your question, then run |
| Combine PDF + image | Upload both, ask a question that spans both sources |

---

## 📊 Output Tabs

After a run completes, results are shown across four tabs:

- **💬 Response** — The final synthesized answer with web source links
- **📊 Metrics** — Accuracy, hallucination rate, completeness scores, and total latency
- **🪵 Execution Logs** — Per-agent log table with status, duration, and a visual timeline bar
- **{ } Logs** — Raw JSON dump of execution logs, evaluation metrics, agent reports, and full pipeline state

---

## 🧩 State Schema

The LangGraph pipeline passes a shared state dictionary between agents. Key fields:

| Field | Type | Description |
|---|---|---|
| `messages` | `list` | Conversation history (LangChain messages) |
| `image_path` | `str` | Temp path to uploaded image (empty if none) |
| `pdf_path` | `str` | Temp path to uploaded PDF (empty if none) |
| `current_task` | `str` | Human-readable label of the current step |
| `used_agents` | `list` | Agents that have completed in order |
| `agent_reports` | `dict` | Per-agent output summaries |
| `execution_logs` | `list` | Structured log entries with timestamps and durations |
| `evaluation_metrics` | `dict` | Accuracy, hallucination rate, completeness, reasoning |
| `final_response` | `str` | The final answer shown to the user |

---

## 🔑 Required API Keys

| Service | Purpose | Get it at |
|---|---|---|
| Google Gemini | LLM + Vision (image understanding) | [aistudio.google.com](https://aistudio.google.com) |
| Tavily | Real-time web search | [tavily.com](https://tavily.com) |

---

## 📦 Key Dependencies

```
streamlit
langgraph
langchain-core
langchain-google-genai
tavily-python
Pillow
pypdf / pdfplumber   # for PDF RAG
```

---

## 🛠 Customisation

**Changing the model** — Update the model name in `config.py`.

**Adding a new agent** — Define the agent function in `agents.py`, register it as a node in `main.py`, and add its name to the flow map node states in `app.py`'s `build_flow_map()`.

**Changing agent names** — If your agents use different internal names than `supervisor`, `pdf_agent`, `image_agent`, `web_agent`, `synthesizer`, `evaluator`, update the `node_state()` calls inside `build_flow_map()` in `app.py` to match.

---

## 📄 License

MIT License. See `LICENSE` for details.
