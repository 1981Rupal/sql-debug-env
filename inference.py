#!/usr/bin/env python3
"""
inference.py — SQL Debug Environment Baseline Agent
=====================================================
Runs an LLM agent against all 9 SQL Debug tasks and emits
structured [START] / [STEP] / [END] logs to stdout.

Required environment variables:
  API_BASE_URL     OpenAI-compatible API endpoint
  MODEL_NAME       Model identifier
  HF_TOKEN         Hugging Face API token (no default)
  ENV_URL          Running environment URL (default: http://localhost:7860)

Optional:
  LOCAL_IMAGE_NAME  Docker image name if using from_docker_image()
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Optional
import requests
from openai import OpenAI

# ─── Config (exact format required by checklist) ──────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")          # no default — must be set by user
API_KEY      = HF_TOKEN or ""
ENV_URL      = os.environ.get("ENV_URL",       "http://localhost:7860")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

TASK_NAME               = "sql-debug"
BENCHMARK               = "sql-debug-env"
MAX_STEPS               = 5
MAX_TOTAL_REWARD        = 9.0    # 9 tasks x max 1.0 each
SUCCESS_SCORE_THRESHOLD = 0.6

TASKS_TO_RUN = [
    "easy_1", "easy_2", "easy_3",
    "medium_1", "medium_2", "medium_3",
    "hard_1", "hard_2", "hard_3",
]

# ─── Structured Logging (exact START/STEP/END format required by judges) ──────

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "event":     "[START]",
        "task":      task,
        "env":       env,
        "model":     model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(json.dumps({
        "event":     "[STEP]",
        "step":      step,
        "action":    action,
        "reward":    reward,
        "done":      done,
        "error":     error,
        "timestamp": time.time(),
    }), flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(json.dumps({
        "event":     "[END]",
        "success":   success,
        "steps":     steps,
        "score":     round(score, 4),
        "rewards":   rewards,
        "timestamp": time.time(),
    }), flush=True)


# ─── Environment HTTP Client ──────────────────────────────────────────────────

def env_reset() -> dict:
    r = requests.post(f"{ENV_URL}/reset", timeout=30)
    r.raise_for_status()
    return r.json()


def env_grade(task_id: str, fixed_query: str) -> dict:
    r = requests.post(
        f"{ENV_URL}/grade/{task_id}",
        json={"fixed_query": fixed_query},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_tasks() -> List[dict]:
    r = requests.get(f"{ENV_URL}/tasks", timeout=30)
    r.raise_for_status()
    return r.json()


# ─── LLM Agent ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SQL developer. Your job is to fix a broken SQL query.\n\n"
    "Rules:\n"
    "1. Return ONLY the corrected SQL query — no explanation, no markdown, no code fences.\n"
    "2. Keep the query intent identical to what is described; only fix the bug(s).\n"
    "3. End the query with a semicolon.\n"
    "4. Do not add or remove columns unless the bug explicitly requires it.\n"
    "If you see a previous error, use it to guide your fix."
)


def get_model_message(
    client: OpenAI,
    step: int,
    task_description: str,
    broken_query: str,
    error_hint: Optional[str],
    last_error: Optional[str],
    last_attempt: Optional[str],
    history: List[str],
) -> str:
    """Ask the LLM to fix the SQL query. Returns the fixed query string."""
    try:
        parts = [
            f"Task: {task_description}",
            f"Broken SQL:\n{broken_query}",
        ]
        if error_hint and step == 1:
            parts.append(f"Hint: {error_hint}")
        if last_attempt:
            parts.append(f"Your previous attempt:\n{last_attempt}")
        if last_error:
            parts.append(f"Error from previous attempt: {last_error}")
        parts.append("Return ONLY the corrected SQL query.")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "\n\n".join(parts)},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```sql", "").replace("```", "").strip()
        return raw

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return broken_query  # fallback: won't crash, scores low


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_tasks = env_tasks()
    task_map  = {t["id"]: t for t in all_tasks}

    all_rewards:    List[float] = []
    total_steps_taken           = 0
    overall_score               = 0.0
    overall_success             = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for task_id in TASKS_TO_RUN:
            task_info = task_map.get(task_id)
            if not task_info:
                print(f"[DEBUG] Task {task_id} not found, skipping.", flush=True)
                continue

            description  = task_info.get("task_description") or task_info.get("description", "")
            broken_query = task_info["broken_query"]
            error_hint   = task_info.get("error_hint")

            history:      List[str]   = []
            rewards:      List[float] = []
            steps_taken               = 0
            task_done                 = False
            last_attempt: Optional[str] = None
            last_error:   Optional[str] = None

            env_reset()

            for step in range(1, MAX_STEPS + 1):
                if task_done:
                    break

                message = get_model_message(
                    client=client,
                    step=step,
                    task_description=description,
                    broken_query=broken_query,
                    error_hint=error_hint,
                    last_error=last_error,
                    last_attempt=last_attempt,
                    history=history,
                )

                grade_result = env_grade(task_id, message)
                reward       = grade_result.get("reward", 0.0)
                done         = grade_result.get("success", False)
                error        = grade_result.get("execution_error")

                rewards.append(reward)
                steps_taken  = step
                last_attempt = message
                last_error   = error or grade_result.get("feedback")
                task_done    = done

                log_step(
                    step=step,
                    action=message,
                    reward=reward,
                    done=done,
                    error=error,
                )

                history.append(f"Step {step}: reward={reward:.2f}")

            task_best = max(rewards) if rewards else 0.0
            all_rewards.append(task_best)
            total_steps_taken += steps_taken

        total_reward    = sum(all_rewards)
        overall_score   = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        overall_score   = min(max(overall_score, 0.0), 1.0)
        overall_success = overall_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(
            success=overall_success,
            steps=total_steps_taken,
            score=overall_score,
            rewards=all_rewards,
        )


if __name__ == "__main__":
    asyncio.run(main())
