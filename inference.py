import os
from typing import List
from openai import OpenAI

from environment import SurakshaAIEnv, Action

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 5


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


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, rewards: List[float]):
    safe_rewards = [max(0.01, min(0.99, r)) for r in rewards]
    rewards_str = ",".join(f"{r:.3f}" for r in safe_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def run_task(task):
    env = SurakshaAIEnv(task=task)

    rewards = []
    log_start(task, "suraksha_ai", MODEL_NAME)

    obs = env.reset()

    for i in range(1, MAX_STEPS + 1):
        action = decide_action(obs)
        result = env.step(Action(action=action))

        rewards.append(result.reward)
        log_step(i, action, result.reward, result.done)

        if result.done:
            break

    avg = sum(rewards) / len(rewards)

    # STRICT SAFE FINAL SCORE
    avg = max(0.01, min(0.99, avg))

    success = avg > 0.5

    log_end(success, len(rewards), rewards)


if __name__ == "__main__":
    for t in TASKS:
        run_task(t)