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
# Clamp function (IMPORTANT FIX)
# -----------------------------
def clamp_score(score):
    return max(0.01, min(score, 0.99))


# -----------------------------
# Email Alert Function 
# -----------------------------
def send_email_alert(receiver_email):
    sender_email = os.getenv("EMAIL_USER")
    app_password = os.getenv("EMAIL_PASS")

    subject = "Emergency Alert!"
    body = "User might be in danger. Immediate attention required!"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        print("Email Sent Successfully!")
        return True
    except Exception as e:
        print("Email Failed:", e)
        return False


# -----------------------------
# Environment Class
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
    # Grader (FIXED)
    # -----------------------------
    def grade_action(self, observation, action):
        if observation.sound == "scream":
            score = 0.99 if action in ["call_police", "send_alert"] else 0.01
        elif observation.movement == "suspicious":
            score = 0.99 if action == "send_alert" else 0.01
        else:
            score = 0.99 if action == "ignore" else 0.01

        return clamp_score(score)

    # -----------------------------
    # Step
    # -----------------------------
    def step(self, action: Action) -> StepResult:
        self.step_count += 1
        obs = self.state

        # Danger score
        danger_score = 0.0
        if obs.sound == "scream":
            danger_score += 0.7
        if obs.movement == "suspicious":
            danger_score += 0.3

        grade = self.grade_action(obs, action.action)
        reward = (danger_score * 0.5) + (grade * 0.5)

        # FINAL CLAMP (IMPORTANT)
        reward = clamp_score(round(reward, 2))

        # Email trigger
        alert_status = False

        if action.action == "send_alert":
            try:
                alert_status = send_email_alert("shivam738804@gmail.com")
            except Exception as e:
                print("Email error:", e)
                alert_status = False  

        done = self.step_count >= self.max_steps

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "task": self.task,
                "danger_score": clamp_score(danger_score),
                "grade": grade,
                "alert_sent": alert_status,
                "alert_message": "Alert triggered"
            }
        )

    def state(self) -> Observation:
        return self.state