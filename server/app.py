"""
SQL Debug Environment — FastAPI Server
Real-world environment where an AI agent fixes broken SQL queries.
"""

import uuid
import sqlite3
import re
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class SQLAction(BaseModel):
    """Action the agent takes: submit a fixed SQL query."""
    fixed_query: str
    task_id: Optional[str] = None


class SQLObservation(BaseModel):
    """What the agent observes after each step."""
    task_id: str
    task_description: str
    broken_query: str
    difficulty: str                # "easy" | "medium" | "hard"
    error_hint: Optional[str]      # syntax hint (easy tasks only)
    execution_result: Optional[str]  # result of agent's query attempt
    execution_error: Optional[str]   # SQL error if query failed
    reward: float
    done: bool
    success: bool
    feedback: str


class EnvState(BaseModel):
    """Episode metadata."""
    episode_id: str
    step_count: int
    current_task_id: Optional[str]
    total_reward: float


# ─── Task Bank ────────────────────────────────────────────────────────────────

TASKS = [
    # ── EASY ──────────────────────────────────────────────────────────────────
    {
        "id": "easy_1",
        "difficulty": "easy",
        "description": "Fix the syntax error: SELECT statement is missing the FROM keyword.",
        "broken_query": "SELECT name, age users WHERE age > 18;",
        "correct_query": "SELECT name, age FROM users WHERE age > 18;",
        "error_hint": "A SELECT query needs FROM to specify the table.",
        "schema": """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER NOT NULL
            );
            INSERT INTO users VALUES (1, 'Alice', 25);
            INSERT INTO users VALUES (2, 'Bob', 17);
            INSERT INTO users VALUES (3, 'Charlie', 30);
        """,
        "expected_rows": [("Alice", 25), ("Charlie", 30)],
        "grader": "exact_rows",
    },
    {
        "id": "easy_2",
        "difficulty": "easy",
        "description": "Fix the typo: WEHRE should be WHERE.",
        "broken_query": "SELECT product_name FROM products WEHRE price < 100;",
        "correct_query": "SELECT product_name FROM products WHERE price < 100;",
        "error_hint": "Check the spelling of the filter keyword.",
        "schema": """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                price REAL NOT NULL
            );
            INSERT INTO products VALUES (1, 'Pen', 5.0);
            INSERT INTO products VALUES (2, 'Laptop', 999.0);
            INSERT INTO products VALUES (3, 'Notebook', 50.0);
        """,
        "expected_rows": [("Pen",), ("Notebook",)],
        "grader": "exact_rows",
    },
    {
        "id": "easy_3",
        "difficulty": "easy",
        "description": "Fix the missing semicolon and wrong keyword: ORDR BY should be ORDER BY.",
        "broken_query": "SELECT name FROM employees ORDR BY salary DESC",
        "correct_query": "SELECT name FROM employees ORDER BY salary DESC;",
        "error_hint": "Check the sorting keyword spelling.",
        "schema": """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                salary REAL NOT NULL
            );
            INSERT INTO employees VALUES (1, 'Alice', 90000);
            INSERT INTO employees VALUES (2, 'Bob', 75000);
            INSERT INTO employees VALUES (3, 'Charlie', 110000);
        """,
        "expected_rows": [("Charlie",), ("Alice",), ("Bob",)],
        "grader": "exact_rows",
    },

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
        "id": "medium_1",
        "difficulty": "medium",
        "description": (
            "Fix the logic error: The query should return employees with salary > 80000, "
            "but currently uses the wrong comparison operator."
        ),
        "broken_query": "SELECT name, salary FROM employees WHERE salary < 80000 ORDER BY salary DESC;",
        "correct_query": "SELECT name, salary FROM employees WHERE salary > 80000 ORDER BY salary DESC;",
        "error_hint": None,
        "schema": """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                salary REAL NOT NULL
            );
            INSERT INTO employees VALUES (1, 'Alice', 90000);
            INSERT INTO employees VALUES (2, 'Bob', 75000);
            INSERT INTO employees VALUES (3, 'Charlie', 110000);
            INSERT INTO employees VALUES (4, 'Diana', 50000);
        """,
        "expected_rows": [("Charlie", 110000.0), ("Alice", 90000.0)],
        "grader": "exact_rows",
    },
    {
        "id": "medium_2",
        "difficulty": "medium",
        "description": (
            "Fix the JOIN: The query joins on the wrong column. "
            "orders.user_id should join to users.id, not users.name."
        ),
        "broken_query": (
            "SELECT users.name, orders.product FROM users "
            "JOIN orders ON users.name = orders.user_id;"
        ),
        "correct_query": (
            "SELECT users.name, orders.product FROM users "
            "JOIN orders ON users.id = orders.user_id;"
        ),
        "error_hint": None,
        "schema": """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                product TEXT NOT NULL
            );
            INSERT INTO users VALUES (1, 'Alice');
            INSERT INTO users VALUES (2, 'Bob');
            INSERT INTO orders VALUES (1, 1, 'Laptop');
            INSERT INTO orders VALUES (2, 2, 'Phone');
            INSERT INTO orders VALUES (3, 1, 'Tablet');
        """,
        "expected_rows": [("Alice", "Laptop"), ("Bob", "Phone"), ("Alice", "Tablet")],
        "grader": "exact_rows_unordered",
    },
    {
        "id": "medium_3",
        "difficulty": "medium",
        "description": (
            "Fix the GROUP BY error: The query groups by name but should group by department "
            "to get total salary per department."
        ),
        "broken_query": (
            "SELECT department, SUM(salary) as total_salary "
            "FROM employees GROUP BY name;"
        ),
        "correct_query": (
            "SELECT department, SUM(salary) as total_salary "
            "FROM employees GROUP BY department;"
        ),
        "error_hint": None,
        "schema": """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL
            );
            INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000);
            INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 60000);
            INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000);
            INSERT INTO employees VALUES (4, 'Diana', 'Marketing', 70000);
        """,
        "expected_rows": [("Engineering", 200000.0), ("Marketing", 130000.0)],
        "grader": "exact_rows_unordered",
    },

    # ── HARD ──────────────────────────────────────────────────────────────────
    {
        "id": "hard_1",
        "difficulty": "hard",
        "description": (
            "Multiple bugs: (1) HAVING clause should filter departments with avg salary > 80000, "
            "not < 80000. (2) Should use AVG not SUM. (3) Results should be ordered by avg_salary DESC."
        ),
        "broken_query": (
            "SELECT department, SUM(salary) as avg_salary "
            "FROM employees GROUP BY department "
            "HAVING SUM(salary) < 80000 ORDER BY avg_salary ASC;"
        ),
        "correct_query": (
            "SELECT department, AVG(salary) as avg_salary "
            "FROM employees GROUP BY department "
            "HAVING AVG(salary) > 80000 ORDER BY avg_salary DESC;"
        ),
        "error_hint": None,
        "schema": """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL
            );
            INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000);
            INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 60000);
            INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000);
            INSERT INTO employees VALUES (4, 'Diana', 'Marketing', 70000);
            INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 85000);
            INSERT INTO employees VALUES (6, 'Frank', 'HR', 55000);
        """,
        "expected_rows": [("Engineering", 95000.0)],
        "grader": "partial_numeric",
    },
    {
        "id": "hard_2",
        "difficulty": "hard",
        "description": (
            "Fix the subquery: The query should find customers who placed MORE than 2 orders, "
            "but the subquery uses COUNT(*) = 1 and the outer query filter is wrong."
        ),
        "broken_query": (
            "SELECT name FROM customers WHERE id IN "
            "(SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) = 1) "
            "ORDER BY name;"
        ),
        "correct_query": (
            "SELECT name FROM customers WHERE id IN "
            "(SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 2) "
            "ORDER BY name;"
        ),
        "error_hint": None,
        "schema": """
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                amount REAL NOT NULL
            );
            INSERT INTO customers VALUES (1, 'Alice');
            INSERT INTO customers VALUES (2, 'Bob');
            INSERT INTO customers VALUES (3, 'Charlie');
            INSERT INTO orders VALUES (1, 1, 100);
            INSERT INTO orders VALUES (2, 1, 200);
            INSERT INTO orders VALUES (3, 1, 150);
            INSERT INTO orders VALUES (4, 2, 300);
            INSERT INTO orders VALUES (5, 3, 50);
            INSERT INTO orders VALUES (6, 3, 75);
        """,
        "expected_rows": [("Alice",)],
        "grader": "exact_rows",
    },
    {
        "id": "hard_3",
        "difficulty": "hard",
        "description": (
            "Fix the CTE (WITH clause): The CTE calculates wrong metric — it uses MAX instead of MIN "
            "for finding each department's lowest salary. Also the outer query filter is inverted."
        ),
        "broken_query": (
            "WITH dept_min AS ("
            "  SELECT department, MAX(salary) as min_salary FROM employees GROUP BY department"
            ") "
            "SELECT e.name, e.department, e.salary FROM employees e "
            "JOIN dept_min d ON e.department = d.department "
            "WHERE e.salary > d.min_salary ORDER BY e.department, e.salary;"
        ),
        "correct_query": (
            "WITH dept_min AS ("
            "  SELECT department, MIN(salary) as min_salary FROM employees GROUP BY department"
            ") "
            "SELECT e.name, e.department, e.salary FROM employees e "
            "JOIN dept_min d ON e.department = d.department "
            "WHERE e.salary > d.min_salary ORDER BY e.department, e.salary;"
        ),
        "error_hint": None,
        "schema": """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL
            );
            INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000);
            INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 70000);
            INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000);
            INSERT INTO employees VALUES (4, 'Diana', 'Marketing', 60000);
            INSERT INTO employees VALUES (5, 'Eve', 'Marketing', 80000);
        """,
        "expected_rows": [
            ("Alice", "Engineering", 90000.0),
            ("Charlie", "Engineering", 110000.0),
            ("Eve", "Marketing", 80000.0),
        ],
        "grader": "partial_numeric",
    },
]

TASK_MAP = {t["id"]: t for t in TASKS}


# ─── Grading Logic ────────────────────────────────────────────────────────────

def run_query_on_schema(schema_sql: str, query: str):
    """Run a query against an in-memory SQLite DB with the given schema."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema_sql)
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        conn.close()
        return None, str(e)


def grade_response(task: dict, agent_query: str) -> tuple[float, str, str, str]:
    """
    Returns (reward: float, feedback: str, exec_result: str, exec_error: str)
    reward is in [0.0, 1.0]
    """
    rows, error = run_query_on_schema(task["schema"], agent_query)

    if error:
        # Partial credit if the query is syntactically close to correct
        partial = 0.1 if _is_structurally_close(task["broken_query"], agent_query) else 0.0
        return partial, f"SQL error: {error}", None, error

    exec_result = str(rows)
    expected = task["expected_rows"]
    grader = task["grader"]

    if grader == "exact_rows":
        if rows == expected:
            return 1.0, "Perfect! Query returns exactly the expected rows.", exec_result, None
        elif set(rows) == set(expected):
            return 0.7, "Rows are correct but in wrong order.", exec_result, None
        else:
            overlap = len(set(rows) & set(expected))
            partial = round(0.3 * overlap / max(len(expected), 1), 2)
            return partial, f"Partially correct. Got {len(rows)} rows, expected {len(expected)}.", exec_result, None

    elif grader == "exact_rows_unordered":
        if set(rows) == set(expected):
            return 1.0, "Perfect! All expected rows returned.", exec_result, None
        else:
            overlap = len(set(rows) & set(expected))
            partial = round(0.5 * overlap / max(len(expected), 1), 2)
            return partial, f"Partial match: {overlap}/{len(expected)} rows correct.", exec_result, None

    elif grader == "partial_numeric":
        if rows == expected:
            return 1.0, "Perfect! Exact match.", exec_result, None
        # Check row-level overlap (unordered, numeric-tolerant)
        expected_set = set(expected)
        row_set = set(rows)
        exact_overlap = len(row_set & expected_set)
        if exact_overlap == len(expected):
            return 0.9, "All rows present, minor ordering issue.", exec_result, None
        if exact_overlap > 0:
            partial = round(0.4 + 0.5 * exact_overlap / len(expected), 2)
            return partial, f"Partial: {exact_overlap}/{len(expected)} rows correct.", exec_result, None
        # Structural credit: query ran, correct number of columns, same column types
        if rows and expected and len(rows[0]) == len(expected[0]):
            # Check if at least the right columns appear (same schema)
            col_match = all(_is_numeric(rows[0][i]) == _is_numeric(expected[0][i])
                           for i in range(len(rows[0])))
            if col_match:
                # Check if the query fixes the primary bug (uses correct function)
                correct_keywords = task.get("correct_query", "").upper()
                structural_ok = all(
                    kw in agent_query.upper()
                    for kw in ["WITH", "MIN", "JOIN"] if kw in correct_keywords
                )
                score = 0.35 if structural_ok else 0.2
                return score, f"Query ran correctly but wrong rows. Got {len(rows)} rows.", exec_result, None
        return 0.15, "Query ran but results don't match expected.", exec_result, None

    return 0.0, "Unknown grader.", exec_result, None


def _is_numeric(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _is_structurally_close(broken: str, agent: str) -> bool:
    """Heuristic: agent query shares most keywords with broken query."""
    def keywords(s):
        return set(re.findall(r'\b[A-Z_]+\b', s.upper()))
    bk, ak = keywords(broken), keywords(agent)
    if not bk:
        return False
    return len(bk & ak) / len(bk) > 0.6


# ─── Environment State ────────────────────────────────────────────────────────

_episode_id = str(uuid.uuid4())
_step_count = 0
_current_task = None
_total_reward = 0.0
_task_queue = list(TASKS)
_task_index = 0


def _pick_next_task():
    global _task_index, _task_queue
    task = _task_queue[_task_index % len(_task_queue)]
    _task_index += 1
    return task


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SQL Debug Environment",
    description=(
        "An OpenEnv-compatible environment where an LLM agent fixes broken SQL queries. "
        "Tasks range from easy syntax errors to hard multi-bug logical errors."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "name": "sql-debug-env",
        "version": "1.0.0",
        "description": "Fix broken SQL queries — easy to hard",
        "endpoints": ["/reset", "/step", "/state", "/tasks"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _do_reset():
    global _episode_id, _step_count, _current_task, _total_reward
    _episode_id = str(uuid.uuid4())
    _step_count = 0
    _total_reward = 0.0
    _current_task = _pick_next_task()
    task = _current_task
    return SQLObservation(
        task_id=task["id"],
        task_description=task["description"],
        broken_query=task["broken_query"],
        difficulty=task["difficulty"],
        error_hint=task.get("error_hint"),
        execution_result=None,
        execution_error=None,
        reward=0.0,
        done=False,
        success=False,
        feedback="Environment reset. Fix the broken SQL query.",
    )


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
        return JSONResponse(
            status_code=400,
            content={"error": "Call /reset first."},
        )

    _step_count += 1
    task = _current_task

    reward, feedback, exec_result, exec_error = grade_response(task, action.fixed_query)
    _total_reward += reward
    done = reward >= 1.0 or _step_count >= 5

    return SQLObservation(
        task_id=task["id"],
        task_description=task["description"],
        broken_query=task["broken_query"],
        difficulty=task["difficulty"],
        error_hint=task.get("error_hint") if reward < 0.5 else None,
        execution_result=exec_result,
        execution_error=exec_error,
        reward=reward,
        done=done,
        success=reward >= 1.0,
        feedback=feedback,
    )


@app.get("/state")
def state():
    return EnvState(
        episode_id=_episode_id,
        step_count=_step_count,
        current_task_id=_current_task["id"] if _current_task else None,
        total_reward=_total_reward,
    )


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata (no answers)."""
    return [
        {
            "id": t["id"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "broken_query": t["broken_query"],
            "error_hint": t.get("error_hint"),
        }
        for t in TASKS
    ]


@app.post("/grade/{task_id}")
def grade_task(task_id: str, action: SQLAction):
    """Grade a specific task directly (for evaluation pipelines)."""
    task = TASK_MAP.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": f"Task {task_id} not found."})
    reward, feedback, exec_result, exec_error = grade_response(task, action.fixed_query)
    return {
        "task_id": task_id,
        "difficulty": task["difficulty"],
        "reward": reward,
        "feedback": feedback,
        "execution_result": exec_result,
        "execution_error": exec_error,
        "success": reward >= 1.0,
    }
