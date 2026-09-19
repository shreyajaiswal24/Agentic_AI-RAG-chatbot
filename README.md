# Agentic AI eBook — RAG Chatbot

A chatbot that answers questions **only** from Konverge AI's eBook
[*Agentic AI: An Executive's Guide*](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).
If the answer is not in the book, it says so instead of guessing.

Built with **Python · LangGraph · Pinecone · Groq LLM · FastAPI**.

---

## 1. Setup

You need Python 3.10+ and two free API keys: [Pinecone](https://app.pinecone.io) and [Groq](https://console.groq.com).

```bash
git clone https://github.com/shreyajaiswal24/Agentic_AI-RAG-chatbot.git
cd Agentic_AI-RAG-chatbot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # open .env and paste your two keys
```

`.env` needs only:

```
PINECONE_API_KEY=pcsk_...
GROQ_API_KEY=gsk_...
```

## 2. Ingest the PDF (run once)

```bash
python -m scripts.ingest
```

Downloads the PDF, cleans and chunks the text, creates the embeddings and stores them in Pinecone. Takes about 30 seconds.

## 3. Start the chatbot

```bash
uvicorn app.main:app --port 8000
```

| Open | What you get |
|---|---|
| http://localhost:8000 | Chat UI in the browser |
| http://localhost:8000/docs | API docs (try `/ask` interactively) |

Or call the API directly:

```bash
curl -s localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What are the core pillars of an Agentic AI system?"}'
```

```json
{
  "answer": "The core pillars are Perception, Reasoning, Planning, Learning, and Execution (Page 17; Page 19).",
  "grounded": true,
  "route": "generate",
  "relevance_score": 0.5968,
  "confidence": { "score": 1.0, "level": "high", "verdict": "fully_supported",
                  "supported_claims": 2, "total_claims": 2, "reason": "..." },
  "retrieved_chunks": [
    { "chunk_id": "p019-c00", "page": 19, "score": 0.5968, "text": "2.1 The Core Pillars: From Perception to Execution ..." },
    { "chunk_id": "p017-c00", "page": 17, "score": 0.5493, "text": "..." }
  ]
}
```

Every response contains the **final answer**, the **retrieved context chunks** (with page numbers), and **two scores**:

- `relevance_score` — cosine similarity between the question and the best chunk (how good the *retrieval* was)
- `confidence` — a second LLM pass fact-checks the answer against the chunks; `score` = supported claims ÷ total claims (how well the *answer* is backed by the book)

---

## 4. Sample queries

Run `python -m scripts.sample_queries` while the server is up, or paste these into the chat UI.

| # | Question | What happens |
|---|---|---|
| 1 | What is Agentic AI and how does it differ from traditional AI tools? | Answered, cites pages 8–11 |
| 2 | What are the core pillars of an Agentic AI system? | Answered, cites page 17 |
| 3 | What challenges do multi-agent systems face and how can they be mitigated? | Answered from several chunks (pages 36, 39, 40) |
| 4 | Which industries does the eBook give Agentic AI use cases for? | Answered, cites pages 16, 54 |
| 5 | What is the capital of France? | **Refused** — not in the book, LLM is never called |
| 6 | Who is the CEO of Konverge AI? | **Refused** — the book mentions Konverge but not its CEO, so the LLM declines instead of guessing |

---

## 5. Architecture

```
INGESTION (once)                       QUESTION TIME (every request)
────────────────                       ─────────────────────────────
PDF                                    question
 │ extract text (pypdf)                  │
 │ clean (headers, page numbers)         ▼
 │ chunk (800 chars, page-aware)     retrieve ──► top-5 chunks from Pinecone
 │ embed (Pinecone llama-text-embed)     │
 ▼                                       ▼
Pinecone index                       check_relevance ──► best score < 0.25 ?
                                         │                       │
                                      generate                 refuse
                                   (Groq LLM answers          (fixed "not in the
                                    only from chunks)          knowledge base" reply)
                                         └──────────┬──────────┘
                                                  grade
                                        (2nd LLM pass fact-checks
                                         the answer → confidence)
                                                    │
                                                  answer + chunks + scores
```

**How it stays grounded**

1. Off-topic questions are stopped by the similarity gate — the LLM is never called.
2. The LLM sees only the retrieved chunks and is instructed to reply *"The information is not available in the provided knowledge base"* if they don't contain the answer.
3. A second LLM pass checks every claim in the answer against the chunks and reports the confidence score.

**The LangGraph workflow** (`app/graph/`): `retrieve → check_relevance → generate | refuse → grade`. The conditional edge after `check_relevance` is what makes the refusal free and deterministic.

**Project layout**

```
app/ingestion/   PDF loading, cleaning, chunking
app/retrieval/   embeddings + Pinecone vector store
app/llm/         LLM client, grounding prompt, answer grader
app/graph/       LangGraph state, nodes, graph
app/api/         FastAPI routes and schemas
app/static/      chat UI
scripts/         ingest.py, sample_queries.py
tests/           20 unit tests (no keys needed) + 5 integration tests
```

---

## 6. Tests

```bash
pytest -q                                    # unit tests, no network
RUN_INTEGRATION=1 pytest -m integration -v   # end-to-end against Pinecone + Groq
```

---

For the full list of environment variables, design decisions (chunk size, threshold calibration, why two scores) and the production roadmap, see [docs/DESIGN.md](docs/DESIGN.md).
