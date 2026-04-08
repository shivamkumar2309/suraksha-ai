import os
from typing import List
from openai import OpenAI

from environment import SurakshaAIEnv, Action

# -----------------------------
# ENV VARIABLES (MANDATORY)
# -----------------------------
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://shivam2309-suraksha-ai.hf.space"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "suraksha-ai-agent"
)

HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------
# OPENAI CLIENT (REQUIRED)
# -----------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# -----------------------------
# SETTINGS
# -----------------------------
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 5


# -----------------------------
# SIMPLE DECISION LOGIC (AGENT)
# -----------------------------
def decide_action(observation):
    if observation.sound == "scream":
        return "call_police"
    elif observation.movement == "suspicious":
        return "send_alert"
    else:
        return "ignore"


# -----------------------------
# LOG FUNCTIONS (STRICT FORMAT)
# -----------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    done_str = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}",
        flush=True
    )


def log_end(success, steps, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = str(success).lower()
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True
    )


# -----------------------------
# MAIN EXECUTION
# -----------------------------
def run_task(task_name):
    env = SurakshaAIEnv(task=task_name)

    rewards = []
    steps = 0
    success = False

    log_start(task_name, "suraksha_ai", MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):

            action_str = decide_action(obs)
            action = Action(action=action_str)

            result = env.step(action)

            reward = result.reward
            done = result.done

            rewards.append(reward)
            steps = step

            log_step(step, action_str, reward, done)

            if done:
                break

        # Success criteria
        avg_reward = sum(rewards) / len(rewards)
        success = avg_reward > 0

    except Exception as e:
        log_step(steps, "error", 0.0, True, str(e))

    finally:
        log_end(success, steps, rewards)


# -----------------------------
# RUN ALL TASKS
# -----------------------------
if __name__ == "__main__":
    for task in TASKS:
        run_task(task)