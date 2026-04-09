from typing import Optional
import random
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
import os

class Observation(BaseModel):
    time: str
    location: str
    sound: str
    movement: str

class Action(BaseModel):
    action: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Optional[dict] = None


# STRICT clamp
def clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


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


class SurakshaAIEnv:

    def __init__(self, task="easy"):
        self.task = task
        self.state = None
        self.step_count = 0
        self.max_steps = 5

    def reset(self):
        self.step_count = 0

        if self.task == "easy":
            self.state = Observation("night", "unsafe", "scream", "suspicious")
        elif self.task == "medium":
            self.state = Observation("night", "unsafe", "normal", "suspicious")
        else:
            self.state = Observation(
                random.choice(["day", "night"]),
                random.choice(["safe", "unsafe"]),
                random.choice(["normal", "scream"]),
                random.choice(["normal", "suspicious"]),
            )

        return self.state

    def grade_action(self, obs, action):
        if obs.sound == "scream":
            return 0.99 if action in ["call_police", "send_alert"] else 0.01
        elif obs.movement == "suspicious":
            return 0.99 if action == "send_alert" else 0.01
        return 0.99 if action == "ignore" else 0.01

    def step(self, action: Action):
        self.step_count += 1
        obs = self.state

        danger = 0.0
        if obs.sound == "scream":
            danger += 0.7
        if obs.movement == "suspicious":
            danger += 0.3

        grade = self.grade_action(obs, action.action)

        # STRICT SAFE VALUES
        danger = clamp(danger)
        grade = clamp(grade)

        reward = (danger + grade) / 2
        reward = clamp(reward)

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
                "danger_score": danger,
                "grade": grade,
                "alert_sent": alert
            }
        )