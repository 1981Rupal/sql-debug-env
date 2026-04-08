#!/usr/bin/env python3
"""
inference.py — SQL Debug Environment Baseline Agent
Aligned with app.py FastAPI Server
"""

import os
import sys
import subprocess
import time
import requests
from openai import OpenAI

# ─── Ensure Dependencies ─────────────────────────────────────────────────────
def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("requests")
_ensure("openai")

# ─── Configuration ───────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_NAME    = "sql-debug"
MAX_STEPS    = 5

# Matches your app.py exactly
TASKS_TO_RUN = [
    "easy_1", "easy_2", "easy_3",
    "medium_1", "medium_2", "medium_3",
    "hard_1", "hard_2", "hard_3"
]

# ─── Structured Logging ──────────────────────────────────────────────────────
def log_start():
    print(f"[START] task={TASK_NAME} env=sql-debug-env model={MODEL_NAME}", flush=True)

def log_step(step, reward, done):
    # Rewards from your app.py are already clamped to 0.01-0.99
    print(f"[STEP] step={step} reward={reward} done={done}", flush=True)

def log_end(success, steps, score):
    # Ensure final score is strictly between 0 and 1
    safe_score = max(0.001, min(0.999, round(score, 4)))
    print(f"[END] task={TASK_NAME} score={safe_score} steps={steps} success={success}", flush=True)

# ─── Agent Main Logic ────────────────────────────────────────────────────────
def main():
    # Initialize OpenAI Client (Synchronous)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Fetch task descriptions from your server
    try:
        resp = requests.get(f"{ENV_URL}/tasks", timeout=10)
        resp.raise_for_status()
        task_map = {t["id"]: t for t in resp.json()}
    except Exception as e:
        print(f"Error connecting to environment at {ENV_URL}: {e}")
        return

    log_start()
    
    task_results = []
    total_steps = 0

    for tid in TASKS_TO_RUN:
        if tid not in task_map:
            continue
            
        task = task_map[tid]
        # Align with your app.py @app.post("/reset")
        requests.post(f"{ENV_URL}/reset") 
        
        best_reward_for_task = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            total_steps += 1
            try:
                # LLM Prompting
                prompt = f"Fix this broken SQL query: {task['broken_query']}\nDescription: {task['description']}\nReturn ONLY the fixed SQL query."
                
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert SQL developer. Return ONLY corrected SQL code. No markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=256,
                    temperature=0.1
                )
                
                # Clean up LLM response
                fixed_sql = completion.choices[0].message.content.strip()
                fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
                
                # Grade using your app.py @app.post("/grade/{task_id}")
                grade_resp = requests.post(
                    f"{ENV_URL}/grade/{tid}", 
                    json={"fixed_query": fixed_sql},
                    timeout=10
                ).json()
                
                reward = grade_resp.get("reward", 0.01)
                done = grade_resp.get("success", False)
                
                # Update best reward for this task
                if reward > best_reward_for_task:
                    best_reward_for_task = reward
                
                log_step(step, reward, done)
                
                if done:
                    break
            except Exception as e:
                # Log minor step failure but keep going
                log_step(step, 0.01, False)
                break
        
        task_results.append(best_reward_for_task)

    # Final Summary Scoring
    if task_results:
        avg_score = sum(task_results) / len(task_results)
    else:
        avg_score = 0.01
        
    log_end(success=(avg_score >= 0.6), steps=total_steps, score=avg_score)

if __name__ == "__main__":
    main()
