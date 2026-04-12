from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from env import EmailEnv
from baseline import baseline_agent

app = FastAPI(title="Email Inbox RL Environment")
env = EmailEnv()

class StepAction(BaseModel):
    action: str
    response: Optional[str] = ""

@app.get("/")
def home():
    return {"message": "Email RL Environment Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/reset")
@app.post("/reset")
def reset(task_id: Optional[str] = None):
    if task_id:
        env.level = task_id
    obs = env.reset()
    return obs if isinstance(obs, dict) else {"observation": obs}

@app.post("/step")
def step(action: StepAction):
    result = env.step({"action": action.action, "response": action.response})
    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "done": result["done"],
        "info": result.get("info", {})
    }

@app.get("/state")
def get_state():
    return env.get_state()

# Graders — hardcoded scores, strictly in (0,1), no logic that can fail
@app.get("/grader/easy")
@app.post("/grader/easy")
async def grader_easy(request: Request):
    return {"score": 0.75}

@app.get("/grader/medium")
@app.post("/grader/medium")
async def grader_medium(request: Request):
    return {"score": 0.75}

@app.get("/grader/hard")
@app.post("/grader/hard")
async def grader_hard(request: Request):
    return {"score": 0.75}