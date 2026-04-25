# MovieMate — LLM-Powered Conversational Movie Recommender

A conversational recommender system (CRS) that uses large language models to suggest movies through natural dialogue. Built on the **ReDial** movie dataset, MovieMate supports three recommendation strategies (Few-Shot, RAG, Agent), real-time voice interaction via ElevenLabs, and an offline evaluation framework with standard IR metrics.

---

## Screenshots

| Voice Mode | Text Mode |
|:---:|:---:|
| ![Voice Mode](Images/image%201.png) | ![Text Mode](https://github.com/nishank03/Conversational-Movies-Recommendation-System/blob/main/Images/Text_Image.png) |

---

## Table of Contents

- [Screenshots](#screenshots)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Recommendation Engines](#recommendation-engines)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Prompt System](#prompt-system)

---

## Features

- **Three recommendation engines** — Few-Shot, RAG (dense + BM25 hybrid), and Agent (ReAct tool-calling loop)
- **Streaming responses** — Server-Sent Events for real-time token-by-token output
- **Voice interaction** — Speech-to-text and text-to-speech via ElevenLabs with word-level highlight sync
- **Multi-provider LLM support** — Anthropic Claude and OpenAI GPT via a unified async client
- **Hybrid retrieval** — FAISS dense search + BM25 lexical search fused with Reciprocal Rank Fusion
- **Offline evaluation** — Hit@K, Recall@K, MRR@K, NDCG@K with comparison reports
- **Prompt ablation** — Three prompt versions (v1/v2/v3) for systematic experimentation
- **Modern UI** — Dark glassmorphism theme with voice and text modes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (UI)                        │
│           Static HTML/CSS/JS • Voice + Text Modes           │
│              SSE streaming • TTS word highlighting          │
└──────────────┬──────────────────────┬───────────────────────┘
               │ /chat/stream (SSE)   │ /voice/* (REST)
               ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Health   │  │   Chat   │  │  Audio   │  │  Streaming │  │
│  │  Routes   │  │  Routes  │  │  Routes  │  │  (SSE)     │  │
│  └──────────┘  └────┬─────┘  └────┬─────┘  └────────────┘  │
│                     │             │                          │
│         ┌───────────┴─────────────┘                         │
│         ▼                                                   │
│  ┌─────────────── Engine Registry ──────────────────┐       │
│  │                                                  │       │
│  │  ┌────────────┐  ┌──────────┐  ┌─────────────┐  │       │
│  │  │  Few-Shot   │  │   RAG    │  │    Agent    │  │       │
│  │  │   Engine    │  │  Engine  │  │   Engine    │  │       │
│  │  └──────┬─────┘  └──┬───┬──┘  └──┬──────────┘  │       │
│  │         │           │   │        │              │       │
│  └─────────┼───────────┼───┼────────┼──────────────┘       │
│            │           │   │        │                       │
│            ▼           ▼   ▼        ▼                       │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────────┐       │
│  │  LLM Client  │ │ Retrieval  │ │ Agent Toolbox    │       │
│  │  (Anthropic  │ │ FAISS+BM25 │ │ (search, lookup, │       │
│  │   / OpenAI)  │ │   + RRF    │ │  user history)   │       │
│  └──────────────┘ └────────────┘ └──────────────────┘       │
│                        │                                    │
│              ┌─────────┴──────────┐                         │
│              │   Data Layer       │                         │
│              │  DatasetLoader     │                         │
│              │  item_map • users  │                         │
│              │  conversations     │                         │
│              └────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘

External Services:
  • Anthropic API / OpenAI API  →  LLM inference
  • ElevenLabs API              →  STT (Scribe v2) + TTS (with timestamps)
  • Sentence-Transformers       →  Local embedding model (all-MiniLM-L6-v2)
```

### Data Flow

1. **User sends a message** (text or voice) via the frontend
2. Voice audio is transcribed via ElevenLabs STT if in voice mode
3. The **Engine Registry** routes to the configured engine (Few-Shot, RAG, or Agent)
4. The engine builds a prompt with context (user profile, retrieval candidates, few-shot examples)
5. The **LLM Client** generates a response with `<REC>item_id1, item_id2</REC>` tags
6. Recommendations are parsed and resolved to movie titles via `item_map`
7. Response streams back as SSE tokens; in voice mode, TTS audio is generated with word-level alignment

---

## Project Structure

```
src/crs/
├── config.py                    # Central settings (pydantic-settings, CRS_ env prefix)
├── schemas.py                   # Shared Pydantic models (Message, Movie, UserProfile, etc.)
├── requirements.txt             # Python dependencies
│
├── api/                         # FastAPI web layer
│   ├── main.py                  # App factory, lifespan, router wiring
│   ├── dependencies.py          # Dependency injection (engine registry, loaders)
│   ├── streaming.py             # SSE stream helpers (<thinking> stripping, <REC> parsing)
│   └── routes/
│       ├── chat.py              # POST /chat/stream (SSE) and POST /chat (JSON)
│       ├── health.py            # GET /healthz and GET /readyz
│       └── audio.py             # POST /voice/transcribe, /voice/speak, /voice/converse
│
├── crs_engines/                 # Recommendation engine strategies
│   ├── base.py                  # Abstract BaseCRS, EngineContext, <REC> parsing
│   ├── few_shot_crs.py          # Prompt-only engine with few-shot examples
│   ├── rag_crs.py               # Retrieval-Augmented Generation (FAISS + BM25 + RRF)
│   └── agent_crs.py             # ReAct tool-calling agent engine
│
├── agents/                      # Tool-calling agent layer
│   ├── orchestrator.py          # ReAct loop with LLM tool calling
│   └── tools.py                 # Tool definitions: search_movies, lookup_movie, get_user_history
│
├── llm/                         # LLM client abstraction
│   ├── client.py                # Anthropic + OpenAI async clients (chat, stream, tool-calling)
│   ├── formatters.py            # History/profile/candidate rendering for prompts
│   └── prompts/
│       ├── system_v1.txt        # Minimal recommender prompt
│       ├── system_v2.txt        # Structured prompt with <REC> format
│       ├── system_v3.txt        # v2 + <thinking> chain-of-thought scaffold
│       └── few_shot_examples.py # Builds few-shot block from training split
│
├── retrieval/                   # Vector and lexical retrieval
│   ├── vector_store.py          # FAISS IndexFlatIP — build, save, load, search
│   ├── embedder.py              # SentenceTransformer wrapper (all-MiniLM-L6-v2)
│   └── bm25.py                  # BM25Okapi index over movie titles
│
├── data/                        # Data loading and processing
│   ├── loaders.py               # DatasetLoader: item_map, user profiles, conversations
│   ├── ingest.py                # Raw → processed parquet pipeline
│   ├── enrich.py                # Mine movie descriptions from conversation text
│   └── split.py                 # Train/eval split generation
│
├── evaluation/                  # Offline evaluation framework
│   ├── metrics.py               # Hit@K, Recall@K, MRR@K, NDCG@K
│   ├── runner.py                # EvaluationRunner — batch eval over dialogue splits
│   └── report.py                # Markdown comparison report renderer
│
├── scripts/
│   └── build_index.py           # CLI: build and save FAISS vector index
│
├── utils/
│   ├── logging.py               # Stdout logging configuration
│   └── timing.py                # Timer context manager
│
└── static/                      # Frontend UI
    ├── index.html               # MovieMate chat interface
    ├── css/style.css             # Dark glassmorphism theme
    └── js/app.js                # SSE chat client, voice recording, TTS playback
```

---

## Recommendation Engines

### 1. Few-Shot Engine (`few_shot`)

The simplest strategy — no retrieval. Constructs a prompt from the system template, user profile, and a block of few-shot examples sampled from the training split. The model recommends based on general knowledge.

- **Always available** (no index required)
- Uses `build_few_shot_block()` with configurable number of examples
- May emit `UNKNOWN` IDs when recommending outside the catalog

### 2. RAG Engine (`rag`)

Retrieval-Augmented Generation with hybrid search:

1. **Query construction** — Combines current message, recent conversation context, and user taste hints
2. **Dense retrieval** — FAISS inner-product search over sentence-transformer embeddings
3. **Lexical retrieval** — BM25 keyword search over movie titles
4. **Fusion** — Reciprocal Rank Fusion (RRF, k=60) merges both ranked lists
5. **Filtering** — Removes movies already in the user's watch history
6. **Generation** — Top candidates are injected into the prompt for grounded recommendations

### 3. Agent Engine (`agent`)

A ReAct-style tool-calling agent that actively searches for movies:

- **Tools available:**
  - `search_movies` — Semantic search over the movie catalog via the vector store
  - `lookup_movie` — Retrieve details for a specific movie by ID
  - `get_user_history` — Fetch the user's watch history and preferences
- **Loop:** LLM decides which tool to call → tool executes → result fed back → repeat (max 4 iterations)
- Recommendations are extracted from the final response using `<REC>` tags

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn, Pydantic v2, pydantic-settings |
| **LLM** | Anthropic Claude / OpenAI GPT (async clients) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Search** | FAISS (IndexFlatIP, cosine via L2-normalized vectors) |
| **Lexical Search** | rank-bm25 (BM25Okapi) |
| **Voice** | ElevenLabs (Scribe v2 STT + Multilingual v2 TTS) |
| **Data** | pandas, numpy |
| **Frontend** | Vanilla JS, Marked.js, Font Awesome, Google Fonts |

---

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for at least one LLM provider (Anthropic or OpenAI)
- ElevenLabs API key (optional, for voice features)

### Installation

```bash
# Clone the repository
git clone https://github.com/nishankelision/Conversational-Movie-Recommendation-System.git
cd Conversational-Movie-Recommendation-System

# Create and activate virtual environment
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r src/crs/requirements.txt
```

### Configuration

Create a `.env` file in `src/crs/`:

```env
CRS_LLM_PROVIDER=openai
CRS_LLM_MODEL=gpt-4o-mini
CRS_OPENAI_API_KEY=sk-...
CRS_ELEVENLABS_API_KEY=sk_...      # Optional, for voice features
CRS_DEFAULT_ENGINE=rag              # few_shot | rag | agent
CRS_PROMPT_VERSION=v3               # v1 | v2 | v3
```

### Build the Vector Index

Required for RAG and Agent engines:

```bash
cd src
python -m crs.scripts.build_index
```

### Run the Server

```bash
cd src
python -m uvicorn crs.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the App

| URL | Description |
|-----|-------------|
| http://localhost:8000 | MovieMate UI (redirects to `/static/index.html`) |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:8000/healthz | Health check |
| http://localhost:8000/readyz | Readiness check (loader, engines, vector store status) |

---

## Configuration

All settings use the `CRS_` prefix and can be set via environment variables or `.env` file.

| Setting | Default | Description |
|---------|---------|-------------|
| `CRS_LLM_PROVIDER` | `anthropic` | LLM provider (`anthropic` or `openai`) |
| `CRS_LLM_MODEL` | `claude-sonnet-4-5` | Model name |
| `CRS_LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `CRS_LLM_MAX_TOKENS` | `1024` | Max output tokens |
| `CRS_DEFAULT_ENGINE` | `rag` | Default recommendation engine |
| `CRS_PROMPT_VERSION` | `v3` | System prompt version (`v1`, `v2`, `v3`) |
| `CRS_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CRS_RETRIEVAL_TOP_K` | `20` | Dense retrieval candidates |
| `CRS_RERANK_TOP_K` | `10` | Final candidates after RRF + filtering |
| `CRS_EVAL_SPLIT_RATIO` | `0.1` | Fraction of data reserved for evaluation |
| `CRS_API_PORT` | `8000` | Server port |

---

## API Reference

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/stream` | SSE streaming response (tokens + recommendations) |
| `POST` | `/chat` | Non-streaming JSON response |

**Request body (`ChatRequest`):**
```json
{
  "message": "I want a mind-bending sci-fi movie",
  "history": [{"role": "user", "content": "..."}],
  "user_id": "user_001",
  "engine": "rag",
  "top_k": 5
}
```

### Voice

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/voice/transcribe` | Audio file → transcribed text (ElevenLabs STT) |
| `POST` | `/voice/speak` | Text → audio with word-level timestamps (TTS) |
| `POST` | `/voice/converse` | Full round-trip: audio → STT → CRS → TTS |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/healthz` | Basic health check |
| `GET` | `/readyz` | Readiness: loader, engines, vector store status |

---

## Evaluation

The evaluation framework replays dialogue history and measures recommendation quality against ground-truth items.

### Metrics

| Metric | Description |
|--------|-------------|
| **Hit@K** | Whether any ground-truth item appears in the top-K predictions |
| **Recall@K** | Fraction of ground-truth items found in the top-K |
| **MRR@K** | Reciprocal rank of the first relevant item |
| **NDCG@K** | Normalized Discounted Cumulative Gain with binary relevance |

Default K values: **1, 3, 5, 10**

### Running Evaluation

```python
from crs.evaluation.runner import EvaluationRunner

runner = EvaluationRunner(engine=engine, loader=loader, settings=settings)
report = await runner.run(eval_records, limit=50)
runner.save_report(report, "results/eval_rag_v3.json")
```

### Comparison Reports

```python
from crs.evaluation.report import render_comparison
markdown = render_comparison([report_rag_v2, report_rag_v3, report_agent_v3])
```

---

## Prompt System

Three prompt versions support ablation studies:

| Version | Features |
|---------|----------|
| **v1** | Minimal — friendly tone, mention titles, no structured output |
| **v2** | Structured — candidate grounding, clarifying questions, `<REC>` output format |
| **v3** | Chain-of-thought — v2 + `<thinking>` scaffold for explicit reasoning before response |

The `<REC>` tag format enables structured extraction of recommended movie IDs:

```
<REC>item_42, item_17, item_5</REC>
```

The `<thinking>` block in v3 is stripped from streamed output so users only see the final response.

---

## License

This project is for academic and research purposes.
