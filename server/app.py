"""
SQL Debug Environment — FastAPI Server
OpenEnv-compatible: step/reset/state + /tasks + /grader + /baseline
All reward scores strictly between 0 and 1 (never 0.0 or 1.0).
"""

import uuid, sqlite3, re
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


def clamp(score: float) -> float:
    """Reward must be STRICTLY between 0 and 1."""
    return round(min(max(float(score), 0.01), 0.99), 4)


# ─── Models ───────────────────────────────────────────────────────────────────

class SQLAction(BaseModel):
    fixed_query: str
    task_id: Optional[str] = None

class SQLObservation(BaseModel):
    task_id: str
    task_description: str
    broken_query: str
    difficulty: str
    error_hint: Optional[str] = None
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None
    reward: float
    done: bool
    success: bool
    feedback: str

class EnvState(BaseModel):
    episode_id: str
    step_count: int
    current_task_id: Optional[str] = None
    total_reward: float


# ─── Tasks ────────────────────────────────────────────────────────────────────
# Three canonical tasks: easy / medium / hard
# Each has a grader function that scores 0.01-0.99

TASKS = [
    {
        "id": "easy",
        "difficulty": "easy",
        "description": "Fix SQL syntax errors — misspelled keywords like WEHRE, ORDR BY, or missing FROM.",
        "broken_query": "SELECT name, age users WHERE age > 18;",
        "correct_query": "SELECT name, age FROM users WHERE age > 18;",
        "error_hint": "Look for misspelled or missing SQL keywords.",
        "schema": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL); INSERT INTO users VALUES (1,'Alice',25),(2,'Bob',17),(3,'Charlie',30);",
        "expected_rows": [("Alice", 25), ("Charlie", 30)],
        "grader": "exact_rows",
    },
    {
        "id": "medium",
        "difficulty": "medium",
        "description": "Fix wrong JOIN column references or incorrect WHERE/GROUP BY logic.",
        "broken_query": "SELECT users.name, orders.product FROM users JOIN orders ON users.name = orders.user_id;",
        "correct_query": "SELECT users.name, orders.product FROM users JOIN orders ON users.id = orders.user_id;",
        "error_hint": None,
        "schema": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL); CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, product TEXT NOT NULL); INSERT INTO users VALUES (1,'Alice'),(2,'Bob'); INSERT INTO orders VALUES (1,1,'Laptop'),(2,2,'Phone'),(3,1,'Tablet');",
        "expected_rows": [("Alice", "Laptop"), ("Bob", "Phone"), ("Alice", "Tablet")],
        "grader": "exact_rows_unordered",
    },
    {
        "id": "hard",
        "difficulty": "hard",
        "description": "Fix wrong aggregation logic — e.g. SUM vs AVG, WHERE vs HAVING, wrong comparison direction.",
        "broken_query": "SELECT department, SUM(salary) as avg_salary FROM employees GROUP BY department HAVING SUM(salary) < 80000 ORDER BY avg_salary ASC;",
        "correct_query": "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department HAVING AVG(salary) > 80000 ORDER BY avg_salary DESC;",
        "error_hint": None,
        "schema": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL, salary REAL NOT NULL); INSERT INTO employees VALUES (1,'Alice','Engineering',90000),(2,'Bob','Marketing',60000),(3,'Charlie','Engineering',110000),(4,'Diana','Marketing',70000),(5,'Eve','Engineering',85000),(6,'Frank','HR',55000);",
        "expected_rows": [("Engineering", 95000.0)],
        "grader": "partial_numeric",
    },
]

TASK_MAP = {t["id"]: t for t in TASKS}


# ─── Grading ──────────────────────────────────────────────────────────────────

def run_query(schema: str, query: str):
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema)
        rows = conn.execute(query).fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        conn.close()
        return None, str(e)


def grade(task: dict, agent_query: str):
    """Returns (reward, feedback, exec_result, exec_error). Reward strictly in (0.01, 0.99)."""
    rows, error = run_query(task["schema"], agent_query)

    if error:
        # Partial credit if structurally close
        kw = lambda s: set(re.findall(r'\b[A-Z_]+\b', s.upper()))
        bk, ak = kw(task["broken_query"]), kw(agent_query)
        close = len(bk & ak) / max(len(bk), 1) > 0.6
        return clamp(0.15 if close else 0.05), f"SQL error: {error}", None, error

    exec_result = str(rows)
    expected = task["expected_rows"]
    grader = task["grader"]

    if grader == "exact_rows":
        if rows == expected:        return clamp(0.95), "Correct! Expected rows returned in order.", exec_result, None
        if set(rows)==set(expected):return clamp(0.75), "Correct rows, wrong order.", exec_result, None
        overlap = len(set(rows) & set(expected))
        return clamp(0.2 + 0.3*overlap/max(len(expected),1)), f"Partial: {overlap}/{len(expected)} rows.", exec_result, None

    elif grader == "exact_rows_unordered":
        if set(rows)==set(expected): return clamp(0.95), "Correct! All expected rows returned.", exec_result, None
        overlap = len(set(rows) & set(expected))
        return clamp(0.2 + 0.4*overlap/max(len(expected),1)), f"Partial: {overlap}/{len(expected)} rows.", exec_result, None

    elif grader == "partial_numeric":
        if rows == expected: return clamp(0.95), "Correct! Exact match.", exec_result, None
        overlap = len(set(rows) & set(expected))
        if overlap: return clamp(0.4 + 0.45*overlap/len(expected)), f"{overlap}/{len(expected)} rows match.", exec_result, None
        if rows and expected and len(rows[0])==len(expected[0]):
            structural = all(kw in agent_query.upper() for kw in ["AVG","HAVING","GROUP"] if kw in task["correct_query"].upper())
            return clamp(0.35 if structural else 0.18), f"Query ran, wrong result ({len(rows)} rows).", exec_result, None
        return clamp(0.12), "Query ran but results don't match.", exec_result, None

    return clamp(0.05), "Unrecognised grader.", exec_result, None


# ─── State ────────────────────────────────────────────────────────────────────

_episode_id   = str(uuid.uuid4())
_step_count   = 0
_current_task = None
_total_reward = 0.0
_task_index   = 0

def _pick():
    global _task_index
    t = TASKS[_task_index % len(TASKS)]
    _task_index += 1
    return t

def _do_reset(task_id: Optional[str] = None):
    global _episode_id, _step_count, _current_task, _total_reward
    _episode_id   = str(uuid.uuid4())
    _step_count   = 0
    _total_reward = 0.0
    _current_task = TASK_MAP[task_id] if task_id and task_id in TASK_MAP else _pick()
    t = _current_task
    return SQLObservation(
        task_id=t["id"], task_description=t["description"],
        broken_query=t["broken_query"], difficulty=t["difficulty"],
        error_hint=t.get("error_hint"), execution_result=None,
        execution_error=None, reward=0.05, done=False, success=False,
        feedback="Environment reset. Fix the broken SQL query.",
    )


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="SQL Debug Environment", version="1.0.0")


@app.get("/")
def root():
    return {"name":"sql-debug-env","version":"1.0.0",
            "description":"Fix broken SQL queries — easy to hard",
            "endpoints":["/reset","/step","/state","/tasks","/grader","/baseline","/health"]}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/reset")
def reset_post(task_id: Optional[str] = None):
    return _do_reset(task_id)

@app.get("/reset")
def reset_get():
    return _do_reset()

@app.post("/step")
def step(action: SQLAction):
    global _step_count, _total_reward, _current_task
    if _current_task is None:
        return JSONResponse(status_code=400, content={"error":"Call /reset first."})
    _step_count += 1
    t = _current_task
    reward, feedback, exec_result, exec_error = grade(t, action.fixed_query)
    _total_reward += reward
    done = reward >= 0.9 or _step_count >= 5
    return SQLObservation(
        task_id=t["id"], task_description=t["description"],
        broken_query=t["broken_query"], difficulty=t["difficulty"],
        error_hint=t.get("error_hint") if reward < 0.5 else None,
        execution_result=exec_result, execution_error=exec_error,
        reward=reward, done=done, success=reward >= 0.9, feedback=feedback,
    )

@app.get("/state")
def state():
    return EnvState(episode_id=_episode_id, step_count=_step_count,
                    current_task_id=_current_task["id"] if _current_task else None,
                    total_reward=_total_reward)

@app.get("/tasks")
def list_tasks():
    """List all tasks — each has a grader."""
    return [
        {
            "id":           t["id"],
            "difficulty":   t["difficulty"],
            "description":  t["description"],
            "broken_query": t["broken_query"],
            "error_hint":   t.get("error_hint"),
            "grader":       t["grader"],
            "has_grader":   True,
            "reward_min":   0.01,
            "reward_max":   0.99,
        }
        for t in TASKS
    ]

@app.post("/grader")
@app.post("/grader/{task_id}")
def grader_endpoint(action: SQLAction, task_id: Optional[str] = None):
    """Grade a query for a specific task without running a full episode."""
    tid  = task_id or action.task_id or (_current_task["id"] if _current_task else None)
    task = TASK_MAP.get(tid) if tid else None
    if not task:
        return JSONResponse(status_code=404, content={"error": f"Task '{tid}' not found. Valid: {list(TASK_MAP)}"})
    reward, feedback, exec_result, exec_error = grade(task, action.fixed_query)
    return {
        "task_id":          task["id"],
        "difficulty":       task["difficulty"],
        "reward":           reward,
        "feedback":         feedback,
        "execution_result": exec_result,
        "execution_error":  exec_error,
        "success":          reward >= 0.9,
    }

# Keep old /grade/{task_id} for backwards compat
@app.post("/grade/{task_id}")
def grade_task(task_id: str, action: SQLAction):
    task = TASK_MAP.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error":f"Task {task_id} not found."})
    reward, feedback, exec_result, exec_error = grade(task, action.fixed_query)
    return {"task_id":task_id,"difficulty":task["difficulty"],"reward":reward,
            "feedback":feedback,"execution_result":exec_result,
            "execution_error":exec_error,"success":reward>=0.9}

@app.get("/baseline")
def baseline():
    """Run oracle baseline — returns score for each task using the correct query."""
    results = []
    for t in TASKS:
        reward, feedback, exec_result, _ = grade(t, t["correct_query"])
        results.append({
            "task_id":    t["id"],
            "difficulty": t["difficulty"],
            "reward":     reward,
            "feedback":   feedback,
            "success":    reward >= 0.9,
        })
    avg = sum(r["reward"] for r in results) / len(results)
    return {"tasks": results, "average_reward": round(avg, 4)}

@app.get("/graders")
def list_graders():
    return [{"task_id":t["id"],"difficulty":t["difficulty"],
             "grader_type":t["grader"],"reward_min":0.01,"reward_max":0.99} for t in TASKS]


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
