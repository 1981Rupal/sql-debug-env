---
title: SQL Debug Environment
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: OpenEnv — LLM agent fixes broken SQL queries (easy to hard)
---

# SQL Debug Environment

**A real-world OpenEnv environment where an LLM agent fixes broken SQL queries.**

Built for the Meta × PyTorch × Scaler OpenEnv Hackathon 2026 (Round 1).

---

## Overview

The SQL Debug Environment presents an AI agent with syntactically or logically broken SQL queries. The agent must identify and fix the bug(s) to produce a query that returns the correct result set against an in-memory SQLite database.

This is a genuine real-world skill: SQL debugging is a core competency for data engineers, analysts, and backend developers. Every task mirrors a real class of production bugs.

---

## Environment Details

| Property | Value |
|---|---|
| Framework | FastAPI + SQLite |
| API | OpenEnv-compatible (reset / step / state) |
| Tasks | 9 (3 easy · 3 medium · 3 hard) |
| Reward range | 0.0 – 1.0 per task |
| Max steps/episode | 5 |
| Infra | vcpu=2, memory=8GB (runs comfortably) |

---

## Task Descriptions

### Easy (syntax errors — detectable from the query text alone)

| ID | Bug |
|---|---|
| `easy_1` | Missing `FROM` keyword |
| `easy_2` | `WEHRE` typo → `WHERE` |
| `easy_3` | `ORDR BY` typo → `ORDER BY` |

### Medium (logic errors — require understanding the data schema)

| ID | Bug |
|---|---|
| `medium_1` | Wrong comparison operator (`<` instead of `>`) |
| `medium_2` | JOIN on wrong column (`users.name` instead of `users.id`) |
| `medium_3` | `GROUP BY` references wrong column |

### Hard (multi-bug — require careful query analysis)

| ID | Bugs |
|---|---|
| `hard_1` | Wrong aggregation (SUM→AVG) + HAVING direction + ORDER direction |
| `hard_2` | Subquery HAVING COUNT condition inverted |
| `hard_3` | CTE uses MAX instead of MIN for minimum salary |

---

## Action Space

```json
{
  "fixed_query": "SELECT name, age FROM users WHERE age > 18;",
  "task_id": "easy_1"
}
```

| Field | Type | Description |
|---|---|---|
| `fixed_query` | string | The agent's corrected SQL query |
| `task_id` | string (optional) | Task identifier (used by `/grade/{task_id}`) |

---

## Observation Space

```json
{
  "task_id": "easy_1",
  "task_description": "Fix the syntax error: SELECT statement is missing the FROM keyword.",
  "broken_query": "SELECT name, age users WHERE age > 18;",
  "difficulty": "easy",
  "error_hint": "A SELECT query needs FROM to specify the table.",
  "execution_result": "[('Alice', 25), ('Charlie', 30)]",
  "execution_error": null,
  "reward": 1.0,
  "done": true,
  "success": true,
  "feedback": "Perfect! Query returns exactly the expected rows."
}
```

---

## Reward Function

| Score | Condition |
|---|---|
| `1.0` | Query returns exactly the expected rows in exact order |
| `0.7` | Correct rows but wrong order |
| `0.3–0.5` | Partial match (some rows correct) |
| `0.1` | Query executes but is structurally close to broken (no correct rows) |
| `0.0` | SQL syntax error or completely wrong output |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Environment info |
| `GET` | `/health` | Health check (returns 200 if live) |
| `POST` | `/reset` | Start new episode, get first task |
| `POST` | `/step` | Submit fixed query, get reward + feedback |
| `GET` | `/state` | Get episode metadata |
| `GET` | `/tasks` | List all 9 tasks |
| `POST` | `/grade/{task_id}` | Grade a specific task directly |

---

## Setup & Run

### Option 1 — Docker (recommended)

```bash
# Build
docker build -t sql-debug-env -f server/Dockerfile .

# Run
docker run -p 7860:7860 sql-debug-env

# Test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
```

### Option 2 — Local Python

```bash
pip install fastapi uvicorn pydantic
cd server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

---

## Running Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_hf_token_here"
export ENV_URL="http://localhost:7860"   # or your HF Space URL

# Install dependencies
pip install openai requests

# Run baseline agent
python inference.py
```

### Expected stdout format

```json
{"event": "[START]", "env_url": "...", "model": "...", "total_tasks": 9, "timestamp": 1234567890.0}
{"event": "[STEP]", "task_id": "easy_1", "step": 1, "difficulty": "easy", "action": "SELECT ...", "reward": 1.0, "done": true, "feedback": "Perfect!", "timestamp": ...}
{"event": "[END]", "total_tasks": 9, "total_reward": 7.8, "avg_reward": 0.867, "results_by_difficulty": {...}, "duration_seconds": 45.2, "timestamp": ...}
```

---

## Pre-submission Checklist

- [x] `openenv.yaml` present and valid
- [x] `server/Dockerfile` builds successfully
- [x] `GET /health` returns 200
- [x] `POST /reset` returns valid observation
- [x] `POST /step` returns reward in [0.0, 1.0]
- [x] `GET /state` returns episode metadata
- [x] 9 tasks with graders (3 easy, 3 medium, 3 hard)
- [x] `inference.py` in root directory
- [x] Uses OpenAI client for all LLM calls
- [x] Emits `[START]` / `[STEP]` / `[END]` stdout logs
- [x] Runtime < 20 minutes on vcpu=2, 8GB RAM
- [x] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` defined

---

## Project Structure

```
sql_debug_env/
├── inference.py          ← Baseline agent (run this to evaluate)
├── openenv.yaml          ← OpenEnv spec
├── README.md             ← This file
└── server/
    ├── app.py            ← FastAPI environment server
    ├── requirements.txt  ← Python dependencies
    └── Dockerfile        ← Container definition
```

---

## License

MIT
