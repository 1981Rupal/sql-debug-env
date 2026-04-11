#!/usr/bin/env python3
"""
inference.py — SQL Debug Environment Baseline Agent
"""

import os, sys, subprocess

# Auto-install missing packages
def _ensure(pkg):
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

_ensure("requests")
_ensure("openai")

import time, asyncio
from typing import List, Optional
import requests
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
# read API_KEY
#  accept HF_TOKEN as fallback so it still works locally.

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")

API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or ""

ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

TASK_NAME               = "sql-debug"
BENCHMARK               = "sql-debug-env"
MAX_STEPS               = 5
MAX_TOTAL_REWARD        = 3.0   # 3 tasks × max ~1.0 each
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS_TO_RUN = ["easy", "medium", "hard"]

# ─── Logging — plain text [START]/[STEP]/[END] as required ───────────────────

# [START], [STEP], [END] 

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ")[:80]
    error_val    = error if error else "null"
    print(f"[STEP] step={step} reward={reward:.4f} done={str(done).lower()} action={action_clean!r} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] task={TASK_NAME} score={score:.4f} steps={steps} success={str(success).lower()} rewards={rewards_str}", flush=True)

# ─── Environment Client ───────────────────────────────────────────────────────
# Connect to our FastAPI server running on HF Spaces.


def env_reset(task_id: Optional[str] = None) -> dict:
    params = {"task_id": task_id} if task_id else {}
    r = requests.post(f"{ENV_URL}/reset", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grade(task_id: str, query: str) -> dict:
    r = requests.post(f"{ENV_URL}/grader/{task_id}",
                      json={"fixed_query": query}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_tasks() -> List[dict]:
    r = requests.get(f"{ENV_URL}/tasks", timeout=30)
    r.raise_for_status()
    return r.json()

# ─── LLM Agent ────────────────────────────────────────────────────────────────

# OpenAI client pointed at whatever URL given.

SYSTEM_PROMPT = (
    "You are an expert SQL developer. Fix the broken SQL query.\n"
    "Rules:\n"
    "1. Return ONLY the corrected SQL — no markdown, no explanation, no code fences.\n"
    "2. Keep the intent identical — only fix the bugs.\n"
    "3. End with a semicolon."
)

def get_fix(client: OpenAI, step: int, description: str,
            broken_query: str, error_hint: Optional[str],
            last_error: Optional[str], last_attempt: Optional[str]) -> str:
    try:
        parts = [f"Task: {description}", f"Broken SQL:\n{broken_query}"]
        if error_hint and step == 1:
            parts.append(f"Hint: {error_hint}")
        if last_attempt:
            parts.append(f"Your previous attempt:\n{last_attempt}")
        if last_error:
            parts.append(f"Error from previous attempt: {last_error}")
        parts.append("Return ONLY the corrected SQL query.")

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "\n\n".join(parts)},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        return raw.replace("```sql", "").replace("```", "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return broken_query   # fallback — won't crash, scores low

# ─── Main Loop ────────────────────────────────────────────────────────────────
# For each task (easy/medium/hard):
#   1. Reset the environment to get the broken query
#   2. Ask LLM to fix it (up to MAX_STEPS tries)
#   3. Grade each attempt
#   4. Log [START], [STEP]s, [END]

async def main() -> None:
   
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_URL={ENV_URL}", flush=True)
    print(f"[DEBUG] API_KEY present={bool(API_KEY)}", flush=True)

    # Fetch task info from our environment
    all_tasks = env_tasks()
    task_map  = {t["id"]: t for t in all_tasks}

    all_rewards:    List[float] = []
    total_steps                 = 0
    overall_success             = False

    for task_id in TASKS_TO_RUN:
        info = task_map.get(task_id)
        if not info:
            print(f"[DEBUG] Task '{task_id}' not found, skipping.", flush=True)
            continue

        desc    = info.get("task_description") or info.get("description", "")
        broken  = info["broken_query"]
        hint    = info.get("error_hint")

        rewards: List[float] = []
        steps_taken           = 0
        task_done             = False
        last_attempt          = None
        last_error            = None

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            env_reset(task_id=task_id)

            for step in range(1, MAX_STEPS + 1):
                if task_done:
                    break

                fixed = get_fix(client, step, desc, broken,
                                hint, last_error, last_attempt)

                result     = env_grade(task_id, fixed)
                reward     = float(result.get("reward", 0.01))
                done       = bool(result.get("success", False))
                exec_error = result.get("execution_error")

                rewards.append(reward)
                steps_taken  = step
                last_attempt = fixed
                last_error   = exec_error or result.get("feedback")
                task_done    = done

                log_step(step=step, action=fixed, reward=reward,
                         done=done, error=exec_error)

        except Exception as e:
            print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

        finally:
            best   = max(rewards) if rewards else 0.01
            best   = min(max(best, 0.01), 0.99)
            s_ok   = best >= 0.9
            log_end(success=s_ok, steps=steps_taken,
                    score=best, rewards=rewards)
            all_rewards.append(best)
            total_steps += steps_taken

    # Overall summary
    overall_score   = sum(all_rewards) / MAX_TOTAL_REWARD if all_rewards else 0.0
    overall_score   = min(max(overall_score, 0.0), 1.0)
    overall_success = overall_score >= SUCCESS_SCORE_THRESHOLD
    print(f"[SUMMARY] total_score={overall_score:.4f} success={overall_success} tasks={len(all_rewards)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
