from fastapi import FastAPI
from environment import SurakshaAIEnv, Action
import uvicorn

app = FastAPI()

env = SurakshaAIEnv(task="easy")


@app.get("/")
def home():
    return {"message": "SurakshaAI Environment Running"}


@app.post("/reset")
def reset(task: str = "easy"):
    global env
    env = SurakshaAIEnv(task=task)
    return env.reset().dict()


@app.post("/step")
def step(action: dict):
    act = Action(**action)
    result = env.step(act)

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }


# IMPORTANT: main function 
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# REQUIRED for OpenEnv
if __name__ == "__main__":
    main()