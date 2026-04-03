from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv

app = FastAPI(title="Email Inbox RL Environment")

env = EmailEnv()

class Action(BaseModel):
    action: str
    response: str

@app.get("/")
def home():
    return {"message": "Email RL Environment Running 🚀"}

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    return env.step(action.dict())

@app.get("/state")
def state():
    return env.state()
