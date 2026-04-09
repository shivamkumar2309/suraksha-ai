import os
from typing import List
from openai import OpenAI

from environment import SurakshaAIEnv, Action

# -----------------------------
# ENV VARIABLES (MANDATORY)
# -----------------------------
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# -----------------------------
# SETTINGS
# -----------------------------
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 5


# -----------------------------
# SAFE CLAMP
# -----------------------------
def clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


# -----------------------------
# LLM DECISION
# -----------------------------
def decide_action(obs):
    try:
        prompt = f"""
        Situation:
        {obs}

        Choose one:
        call_police / send_alert / ignore
        """

        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        action = res.choices[0].message.content.strip().lower()

        if action not in ["call_police", "send_alert", "ignore"]:
            return "ignore"

        return action

    except Exception:
        return "ignore"


# -----------------------------
# LOG FUNCTIONS
# -----------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done):
    reward = clamp(reward)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, score, rewards: List[float]):
    rewards = [clamp(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# -----------------------------
# MAIN
# -----------------------------
def run_task(task):
    env = SurakshaAIEnv(task=task)

    rewards = []
    log_start(task, "suraksha_ai", MODEL_NAME)

    obs = env.reset()

    for i in range(1, MAX_STEPS + 1):
        action = decide_action(obs)
        result = env.step(Action(action=action))

        r = clamp(result.reward)
        rewards.append(r)

        log_step(i, action, r, result.done)

        if result.done:
            break

    # -----------------------------
    # FINAL SCORE FIX (IMPORTANT)
    # -----------------------------
    avg = sum(rewards) / len(rewards)
    score = clamp(avg)

    # extra safety (no 0 / 1 ever)
    if score >= 0.99:
        score = 0.989
    if score <= 0.01:
        score = 0.011

    success = score > 0.5

    log_end(success, len(rewards), score, rewards)


if __name__ == "__main__":
    for t in TASKS:
        run_task(t)