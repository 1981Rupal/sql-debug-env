"""
SQL Debug Environment — FastAPI Server
Real-world OpenEnv environment where an AI agent fixes broken SQL queries.
IMPORTANT: All reward scores are strictly in (0.01, 0.99) — never 0.0 or 1.0.
"""

import uuid
import sqlite3
import re
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


# ─── Reward clamping — scores must be STRICTLY between 0 and 1 ───────────────

def clamp(score: float) -> float:
    """Ensure reward is strictly in (0.01, 0.99), never exactly 0.0 or 1.0."""
    return round(min(max(score, 0.01), 0.99), 4)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

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


# ─── Task Bank ────────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "easy_1", "difficulty": "easy",
        "description": "Fix the syntax error: SELECT statement is missing the FROM keyword.",
        "broken_query": "SELECT name, age users WHERE age > 18;",
        "correct_query": "SELECT name, age FROM users WHERE age > 18;",
        "error_hint": "A SELECT query needs FROM to specify the table.",
        "schema": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL); INSERT INTO users VALUES (1, 'Alice', 25); INSERT INTO users VALUES (2, 'Bob', 17); INSERT INTO users VALUES (3, 'Charlie', 30);",
        "expected_rows": [("Alice", 25), ("Charlie", 30)],
        "grader": "exact_rows",
    },
    {
        "id": "easy_2", "difficulty": "easy",
        "description": "Fix the typo: WEHRE should be WHERE.",
        "broken_query": "SELECT product_name FROM products WEHRE price < 100;",
        "correct_query": "SELECT product_name FROM products WHERE price < 100;",
        "error_hint": "Check the spelling of the filter keyword.",
        "schema": "CREATE TABLE products (id INTEGER PRIMARY KEY, product_name TEXT NOT NULL, price REAL NOT NULL); INSERT INTO products VALUES (1, 'Pen', 5.0); INSERT INTO products VALUES (2, 'Laptop', 999.0); INSERT INTO products VALUES (3, 'Notebook', 50.0);",
        "expected_rows": [("Pen",), ("Notebook",)],
        "grader": "exact_rows",
    },
    {
        "id": "easy_3", "difficulty": "easy",
        "description": "Fix the wrong keyword: ORDR BY should be ORDER BY.",
        "broken_query": "SELECT name FROM employees ORDR BY salary DESC",
        "correct_query": "SELECT name FROM employees ORDER BY salary DESC;",
        "error_hint": "Check the sorting keyword spelling.",
        "schema": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, salary REAL NOT NULL); INSERT INTO employees VALUES (1, 'Alice', 90000); INSERT INTO employees VALUES (2, 'Bob', 75000); INSERT INTO employees VALUES (3, 'Charlie', 110000);",
        "expected_rows": [("Charlie",), ("Alice",), ("Bob",)],
        "grader": "exact_rows",
    },
    {
        "id": "medium_1", "difficulty": "medium",
        "description": "Fix the logic error: query should return employees with salary > 80000 but uses wrong comparison operator.",
        "broken_query": "SELECT name, salary FROM employees WHERE salary < 80000 ORDER BY salary DESC;",
        "correct_query": "SELECT name, salary FROM employees WHERE salary > 80000 ORDER BY salary DESC;",
        "error_hint": None,
        "schema": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, salary REAL NOT NULL); INSERT INTO employees VALUES (1, 'Alice', 90000); INSERT INTO employees VALUES (2, 'Bob', 75000); INSERT INTO employees VALUES (3, 'Charlie', 110000); INSERT INTO employees VALUES (4, 'Diana', 50000);",
        "expected_rows": [("Charlie", 110000.0), ("Alice", 90000.0)],
        "grader": "exact_rows",
    },
    {
        "id": "medium_2", "difficulty": "medium",
        "description": "Fix the JOIN: orders.user_id should join to users.id, not users.name.",
        "broken_query": "SELECT users.name, orders.product FROM users JOIN orders ON users.name = orders.user_id;",
        "correct_query": "SELECT users.name, orders.product FROM users JOIN orders ON users.id = orders.user_id;",
        "error_hint": None,
        "schema": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL); CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, product TEXT NOT NULL); INSERT INTO users VALUES (1, 'Alice'); INSERT INTO users VALUES (2, 'Bob'); INSERT INTO orders VALUES (1, 1, 'Laptop'); INSERT INTO orders VALUES (2, 2, 'Phone'); INSERT INTO orders VALUES (3, 1, 'Tablet');",
        "expected_rows": [("Alice", "Laptop"), ("Bob", "Phone"), ("Alice", "Tablet")],
        "grader": "exact_rows_unordered",
    },
    {
        "id": "medium_3", "difficulty": "medium",
        "description": "Fix the GROUP BY error: should group by department, not name.",
        "broken_query": "SELECT department, SUM(salary) as total_salary FROM employees GROUP BY name;",
        "correct_query": "SELECT department, SUM(salary) as total_salary FROM employees GROUP BY department;",
        "error_hint": None,
        "schema": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL, salary REAL NOT NULL); INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000); INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 60000); INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000); INSERT INTO employees VALUES (4, 'Diana', 'Marketing', 70000);",
        "expected_rows": [("Engineering", 200000.0), ("Marketing", 130000.0)],
        "grader": "exact_rows_unordered",
    },
    {
        "id": "hard_1", "difficulty": "hard",
        "description": "Multiple bugs: use AVG not SUM, HAVING avg > 80000 not < 80000, ORDER BY DESC.",
        "broken_query": "SELECT department, SUM(salary) as avg_salary FROM employees GROUP BY department HAVING SUM(salary) < 80000 ORDER BY avg_salary ASC;",
        "correct_query": "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department HAVING AVG(salary) > 80000 ORDER BY avg_salary DESC;",
        "error_hint": None,
        "schema": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL, salary REAL NOT NULL); INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000); INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 60000); INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000); INSERT INTO employees VALUES (4, 'Diana', 'Marketing', 70000); INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 85000); INSERT INTO employees VALUES (6, 'Frank', 'HR', 55000);",
        "expected_rows": [("Engineering", 95000.0)],
        "grader": "partial_numeric",
    },
    {
        "id": "hard_2", "difficulty": "hard",
        "description": "Fix the subquery: find customers with MORE than 2 orders — HAVING COUNT(*) = 1 is wrong.",
        "broken_query": "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) = 1) ORDER BY name;",
        "correct_query": "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 2) ORDER BY name;",
        "error_hint": None,
        "schema": "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL); CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL, amount REAL NOT NULL); INSERT INTO customers VALUES (1, 'Alice'); INSERT INTO customers VALUES (2, 'Bob'); INSERT INTO customers VALUES (3, 'Charlie'); INSERT INTO orders VALUES (1, 1, 100); INSERT INTO orders VALUES (2, 1, 200); INSERT INTO orders VALUES (3, 1, 150); INSERT INTO orders VALUES (4, 2, 300); INSERT INTO orders VALUES (5, 3, 50); INSERT INTO orders VALUES (6, 3, 75);",
        "expected_rows": [("Alice",)],
        "grader": "exact_rows",
    },
    {
        "id": "hard_3", "difficulty": "hard",
        "description": "Fix the CTE: uses MAX instead of MIN for minimum salary. Find employees ABOVE the minimum.",
        "broken_query": "WITH dept_min AS (SELECT department, MAX(salary) as min_salary FROM employees GROUP BY department) SELECT e.name, e.department, e.salary FROM employees e JOIN dept_min d ON e.department = d.department WHERE e.salary > d.min_salary ORDER BY e.department, e.salary;",
        "correct_query": "WITH dept_min AS (SELECT department, MIN(salary) as min_salary FROM employees GROUP BY department) SELECT e.name, e.department, e.salary FROM employees e JOIN dept_min d ON e.department = d.department WHERE e.salary > d.min_salary ORDER BY e.department, e.salary;",
        "error_hint": None,
        "schema": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL, salary REAL NOT NULL); INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000); INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 70000); INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000); INSERT INTO employees VALUES (4, 'Diana', 'Marketing', 60000); INSERT INTO employees VALUES (5, 'Eve', 'Marketing', 80000);",
        "expected_rows": [("Alice", "Engineering", 90000.0), ("Charlie", "Engineering", 110000.0), ("Eve", "Marketing", 80000.0)],
        "grader": "partial_numeric",
    },
]

TASK_MAP = {t["id"]: t for t in TASKS}


# ─── Grading Logic ────────────────────────────────────────────────────────────

def run_query_on_schema(schema_sql: str, query: str):
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema_sql)
        rows = conn.execute(query).fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        conn.close()
        return None, str(e)


def _is_numeric(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _is_structurally_close(broken: str, agent: str) -> bool:
    def kw(s):
        return set(re.findall(r'\b[A-Z_]+\b', s.upper()))
    bk, ak = kw(broken), kw(agent)
    return len(bk & ak) / max(len(bk), 1) > 0.6


def grade_response(task: dict, agent_query: str):
    """Grade agent query. Returns reward STRICTLY in (0.01, 0.99)."""
    rows, error = run_query_on_schema(task["schema"], agent_query)

    if error:
        score = 0.15 if _is_structurally_close(task["broken_query"], agent_query) else 0.05
        return clamp(score), f"SQL error: {error}", None, error

    exec_result = str(rows)
    expected = task["expected_rows"]
    grader = task["grader"]

    if grader == "exact_rows":
        if rows == expected:
            return clamp(0.95), "Excellent! Query returns the expected rows.", exec_result, None
        elif set(rows) == set(expected):
            return clamp(0.75), "Rows correct but wrong order.", exec_result, None
        else:
            overlap = len(set(rows) & set(expected))
            score = 0.25 + 0.3 * overlap / max(len(expected), 1)
            return clamp(score), f"Partial: {overlap}/{len(expected)} rows correct.", exec_result, None

    elif grader == "exact_rows_unordered":
        if set(rows) == set(expected):
            return clamp(0.95), "Excellent! All expected rows returned.", exec_result, None
        else:
            overlap = len(set(rows) & set(expected))
            score = 0.25 + 0.4 * overlap / max(len(expected), 1)
            return clamp(score), f"Partial: {overlap}/{len(expected)} rows correct.", exec_result, None

    elif grader == "partial_numeric":
        if rows == expected:
            return clamp(0.95), "Excellent! Exact match.", exec_result, None
        row_set, exp_set = set(rows), set(expected)
        exact_overlap = len(row_set & exp_set)
        if exact_overlap == len(expected):
            return clamp(0.88), "All rows present, minor ordering issue.", exec_result, None
        if exact_overlap > 0:
            score = 0.4 + 0.45 * exact_overlap / len(expected)
            return clamp(score), f"Partial: {exact_overlap}/{len(expected)} rows match.", exec_result, None
        if rows and expected and len(rows[0]) == len(expected[0]):
            col_match = all(_is_numeric(rows[0][i]) == _is_numeric(expected[0][i]) for i in range(len(rows[0])))
            if col_match:
                correct_kw = task.get("correct_query", "").upper()
                structural_ok = all(kw in agent_query.upper() for kw in ["WITH", "MIN", "JOIN"] if kw in correct_kw)
                score = 0.35 if structural_ok else 0.18
                return clamp(score), f"Query ran but wrong rows ({len(rows)} returned).", exec_result, None
        return clamp(0.12), "Query ran but results do not match.", exec_result, None

    return clamp(0.05), "Unknown grader.", exec_result, None


# ─── Episode State ────────────────────────────────────────────────────────────

_episode_id = str(uuid.uuid4())
_step_count = 0
_current_task = None
_total_reward = 0.0
_task_index = 0


def _pick_next_task():
    global _task_index
    task = TASKS[_task_index % len(TASKS)]
    _task_index += 1
    return task


def _do_reset():
    global _episode_id, _step_count, _current_task, _total_reward
    _episode_id = str(uuid.uuid4())
    _step_count = 0
    _total_reward = 0.0
    _current_task = _pick_next_task()
    task = _current_task
    return SQLObservation(
        task_id=task["id"], task_description=task["description"],
        broken_query=task["broken_query"], difficulty=task["difficulty"],
        error_hint=task.get("error_hint"), execution_result=None,
        execution_error=None, reward=0.05, done=False, success=False,
        feedback="Environment reset. Fix the broken SQL query.",
    )


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="SQL Debug Environment", version="1.0.0",
              description="OpenEnv environment: LLM agent fixes broken SQL queries.")


@app.get("/")
def root():
    return {"name": "sql-debug-env", "version": "1.0.0",
            "description": "Fix broken SQL queries — easy to hard",
            "endpoints": ["/reset", "/step", "/state", "/tasks"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset_post():
    return _do_reset()


@app.get("/reset")
def reset_get():
    return _do_reset()


@app.post("/step")
def step(action: SQLAction):
    global _step_count, _total_reward, _current_task
    if _current_task is None:
        return JSONResponse(status_code=400, content={"error": "Call /reset first."})
    _step_count += 1
    task = _current_task
    reward, feedback, exec_result, exec_error = grade_response(task, action.fixed_query)
    _total_reward += reward
    done = reward >= 0.9 or _step_count >= 5
    return SQLObservation(
        task_id=task["id"], task_description=task["description"],
        broken_query=task["broken_query"], difficulty=task["difficulty"],
        error_hint=task.get("error_hint") if reward < 0.5 else None,
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
    return [{"id": t["id"], "difficulty": t["difficulty"],
             "description": t["description"], "broken_query": t["broken_query"],
             "error_hint": t.get("error_hint")} for t in TASKS]


@app.post("/grade/{task_id}")
def grade_task(task_id: str, action: SQLAction):
    task = TASK_MAP.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": f"Task {task_id} not found."})
    reward, feedback, exec_result, exec_error = grade_response(task, action.fixed_query)
    return {"task_id": task_id, "difficulty": task["difficulty"], "reward": reward,
            "feedback": feedback, "execution_result": exec_result,
            "execution_error": exec_error, "success": reward >= 0.9}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
