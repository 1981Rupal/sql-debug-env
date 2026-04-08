#!/usr/bin/env python3
"""
inference.py — SQL Debug Environment Baseline Agent
FINAL FIX: Adjusts scores to be within (0, 1) range per validator rules.
"""

import os
import sys
import subprocess

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("requests")
_ensure("openai")

import json
import time
import asyncio
from typing import List, Optional
import requests
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE_URL     = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.environ.get("HF_TOKEN")
API_KEY          = HF_TOKEN or ""
ENV_URL          = os.environ.get("ENV_URL",       "http://localhost:7860")

TASK_NAME               = "sql-debug"
BENCHMARK               = "sql-debug-env"
MAX_STEPS               = 5
MAX_TOTAL_REWARD        = 9.0

TASKS_TO_RUN = [
    "easy_1", "easy_2", "easy_3",
    "medium_1", "medium_2", "medium_3",
    "hard_1", "hard_2", "hard_3",
]

# ─── Structured Logging ───────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool) -> None:
    # Ensure reward isn't exactly 0 or 1 for individual steps if needed
    safe_reward = max(0.01, min(0.99, reward))
    print(f"[STEP] step={step} reward={safe_reward} done={done}", flush=True)

def log_end(success: bool, steps: int, score: float) -> None:
    # CRITICAL FIX: Force score into (0, 1) range (e.g., 0.999 instead of 1.0)
    # This addresses the "One or more task scores are out of range" error.
    safe_score = max(0.001, min(0.999, score))
    print(f"[END] task={TASK_NAME} score={safe_score} steps={steps} success={success}", flush=True)

# ─── Environment HTTP Client ──────────────────────────────────────────────────

def env_reset():
    return requests.post(f"{ENV_URL}/reset", timeout=30).json()

def env_grade(task_id: str, fixed_query: str):
    return requests.post(f"{ENV_URL}/grade/{task_id}", json={"fixed_query": fixed_query}, timeout=30).json()

def env_tasks():
    return requests.get(f"{ENV_URL}/tasks", timeout=30).json()

# ─── LLM Agent ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are an expert SQL developer. Return ONLY the corrected SQL query ending with a semicolon."

def get_model_message(client, task_description, broken_query, last_error, last_attempt):
    try:
        prompt = f"Task: {task_description}\nBroken SQL: {broken_query}"
        if last_error: prompt += f"\nPrev Error: {last_error}\nPrev Attempt: {last_attempt}"
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.1,
        )
        return response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
    except:
        return broken_query

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    try:
        all_tasks = env_tasks()
        task_map = {t["id"]: t for t in all_tasks}
    except:
        return

    total_rewards = []
    total_steps = 0
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    for task_id in TASKS_TO_RUN:
        task_info = task_map.get(task_id)
        if not task_info: continue
        
        env_reset()
        last_attempt, last_error = None, None
        
        for step in range(1, MAX_STEPS + 1):
            query = get_model_message(client, task_info['task_description'], task_info['broken_query'], last_error, last_attempt)
            res = env_grade(task_id, query)
            
            reward = res.get("reward", 0.0)
            done = res.get("success", False)
            
            log_step(step, query[:20], reward, done)
            
            last_attempt, last_error = query, res.get("execution_error") or res.get("feedback")
            total_steps += 1
            if done:
                total_rewards.append(reward)
                break
        else:
            total_rewards.append(0.0)

    # Calculate overall score and clamp it
    final_score = sum(total_rewards) / len(TASKS_TO_RUN) if TASKS_TO_RUN else 0.0
    log_end(success=(final_score > 0.5), steps=total_steps, score=final_score)

if __name__ == "__main__":
    asyncio.run(main())
