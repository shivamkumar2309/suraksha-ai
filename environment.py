from typing import Optional
import random
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
import os

# -----------------------------
# Observation Model
# -----------------------------
class Observation(BaseModel):
    time: str
    location: str
    sound: str
    movement: str


# -----------------------------
# Action Model
# -----------------------------
class Action(BaseModel):
    action: str


# -----------------------------
# Step Result Model
# -----------------------------
class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Optional[dict] = None


# -----------------------------
# Clamp (STRICT SAFE)
# -----------------------------
def clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


# -----------------------------
# Email Alert
# -----------------------------
def send_email_alert(receiver_email):
    try:
        sender_email = os.getenv("EMAIL_USER")
        app_password = os.getenv("EMAIL_PASS")

        msg = MIMEText("Emergency alert triggered")
        msg["Subject"] = "Alert"
        msg["From"] = sender_email
        msg["To"] = receiver_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception:
        return False


# -----------------------------
# Environment
# -----------------------------
class SurakshaAIEnv:

    def __init__(self, task="easy"):
        self.task = task
        self.state = None
        self.step_count = 0
        self.max_steps = 5

    def reset(self) -> Observation:
        self.step_count = 0

        if self.task == "easy":
            self.state = Observation(
                time="night",
                location="unsafe",
                sound="scream",
                movement="suspicious"
            )
        elif self.task == "medium":
            self.state = Observation(
                time="night",
                location="unsafe",
                sound="normal",
                movement="suspicious"
            )
        else:
            self.state = Observation(
                time=random.choice(["day", "night"]),
                location=random.choice(["safe", "unsafe"]),
                sound=random.choice(["normal", "scream"]),
                movement=random.choice(["normal", "suspicious"])
            )

        return self.state

    # -----------------------------
    # Grader (STRICT SAFE)
    # -----------------------------
    def grade_action(self, obs, action):
        if obs.sound == "scream":
            score = 0.99 if action in ["call_police", "send_alert"] else 0.01
        elif obs.movement == "suspicious":
            score = 0.99 if action == "send_alert" else 0.01
        else:
            score = 0.99 if action == "ignore" else 0.01

        return clamp(score)

    # -----------------------------
    # Step (FINAL SAFE)
    # -----------------------------
    def step(self, action: Action) -> StepResult:
        self.step_count += 1
        obs = self.state

        # Raw danger score
        danger = 0.0
        if obs.sound == "scream":
            danger += 0.7
        if obs.movement == "suspicious":
            danger += 0.3

        # Safe values
        safe_danger = clamp(danger)
        safe_grade = self.grade_action(obs, action.action)

        # Reward
        reward = (safe_danger + safe_grade) / 2
        reward = clamp(reward)

        # Email trigger
        alert = False
        if action.action == "send_alert":
            alert = send_email_alert("shivam738804@gmail.com")

        done = self.step_count >= self.max_steps

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "task": self.task,
                "danger_score": safe_danger,
                "grade": safe_grade,
                "alert_sent": alert
            }
        )

    def get_state(self) -> Observation:
        return self.state