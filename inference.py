#!/usr/bin/env python3
"""
inference.py — SQL Debug Environment Baseline Agent
Emits structured JSON [START]/[STEP]/[END] logs to stdout.

Required env vars:
  API_BASE_URL, MODEL_NAME, HF_TOKEN, ENV_URL
"""

import os, sys, subprocess

def _ensure(pkg):
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

_ensure("requests")
_ensure("openai")

import json, time, asyncio
from typing import List, Optional
import requests
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE_URL     = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.environ.get("HF_TOKEN")
API_KEY          = HF_TOKEN or ""
ENV_URL          = os.environ.get("ENV_URL",       "http://localhost:7860")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

TASK_NAME               = "sql-debug"
BENCHMARK               = "sql-debug-env"
MAX_STEPS               = 5
MAX_TOTAL_REWARD        = 9.0
SUCCESS_SCORE_THRESHOLD = 0.6

TASKS_TO_RUN = [
    "easy_1","easy_2","easy_3",
    "medium_1","medium_2","medium_3",
    "hard_1","hard_2","hard_3",
]

# ─── Logging — MUST be JSON, MUST use [START]/[STEP]/[END] ───────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({"event":"[START]","task":task,"env":env,
                      "model":model,"timestamp":time.time()}), flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(json.dumps({"event":"[STEP]","step":step,"action":action,
                      "reward":reward,"done":done,"error":error,
                      "timestamp":time.time()}), flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(json.dumps({"event":"[END]","success":success,"steps":steps,
                      "score":round(score,4),"rewards":rewards,
                      "timestamp":time.time()}), flush=True)

# ─── Env Client ───────────────────────────────────────────────────────────────

def env_reset() -> dict:
    r = requests.post(f"{ENV_URL}/reset", timeout=30)
    r.raise_for_status(); return r.json()

def env_grade(task_id: str, query: str) -> dict:
    r = requests.post(f"{ENV_URL}/grade/{task_id}",
                      json={"fixed_query": query}, timeout=30)
    r.raise_for_status(); return r.json()

def env_tasks() -> List[dict]:
    r = requests.get(f"{ENV_URL}/tasks", timeout=30)
    r.raise_for_status(); return r.json()

# ─── LLM Agent ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SQL developer. Fix the broken SQL query.\n"
    "Rules: 1. Return ONLY the corrected SQL — no markdown, no explanation.\n"
    "2. Keep intent identical; only fix bugs. 3. End with semicolon."
)

def get_model_message(client, step, description, broken_query,
                      error_hint, last_error, last_attempt, history):
    try:
        parts = [f"Task: {description}", f"Broken SQL:\n{broken_query}"]
        if error_hint and step == 1: parts.append(f"Hint: {error_hint}")
        if last_attempt: parts.append(f"Previous attempt:\n{last_attempt}")
        if last_error:   parts.append(f"Error: {last_error}")
        parts.append("Return ONLY the corrected SQL query.")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":"\n\n".join(parts)}],
            max_tokens=512, temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        return raw.replace("```sql","").replace("```","").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return broken_query

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_tasks = env_tasks()
    task_map  = {t["id"]: t for t in all_tasks}
    all_rewards: List[float] = []
    total_steps = 0
    overall_score = 0.0
    overall_success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for task_id in TASKS_TO_RUN:
            info = task_map.get(task_id)
            if not info: continue
            desc    = info.get("task_description") or info.get("description","")
            broken  = info["broken_query"]
            hint    = info.get("error_hint")
            rewards: List[float] = []
            steps_taken = 0
            done = False
            last_attempt = last_error = None
            env_reset()

            for step in range(1, MAX_STEPS + 1):
                if done: break
                msg = get_model_message(client, step, desc, broken,
                                        hint, last_error, last_attempt, [])
                result  = env_grade(task_id, msg)
                reward  = result.get("reward", 0.01)
                done    = result.get("success", False)
                error   = result.get("execution_error")
                rewards.append(reward)
                steps_taken  = step
                last_attempt = msg
                last_error   = error or result.get("feedback")
                log_step(step=step, action=msg, reward=reward, done=done, error=error)

            all_rewards.append(max(rewards) if rewards else 0.01)
            total_steps += steps_taken

        total = sum(all_rewards)
        overall_score   = min(max(total / MAX_TOTAL_REWARD, 0.0), 1.0)
        overall_success = overall_score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=overall_success, steps=total_steps,
                score=overall_score, rewards=all_rewards)

if __name__ == "__main__":
    asyncio.run(main())
