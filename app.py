from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv
from baseline import baseline_agent

app = FastAPI(title="Email Inbox RL Environment")

# Initialize environment
env = EmailEnv()

# -----------------------------
# Request Models
# -----------------------------
class Action(BaseModel):
    action: str
    response: str


# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def home():
    return {"message": "Email RL Environment Running 🚀"}


# -----------------------------
# RESET (CRITICAL FIX)
# -----------------------------
# REQUIRED for validator
@app.post("/reset")
def reset_post():
    obs = env.reset()
    return obs["observation"] if "observation" in obs else obs

@app.get("/reset")
def reset_get():
    obs = env.reset()
    return obs["observation"] if "observation" in obs else obs


# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
def step(action: Action):
    result = env.step({
        "action": action.action,
        "response": action.response
    })

    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"]
    }


# -----------------------------
# STATE (REQUIRED)
# -----------------------------
@app.get("/state")
def get_state():
    return env.get_state()


# -----------------------------
# BASELINE
# -----------------------------
@app.get("/baseline")
def baseline():
    obs = env.get_state()
    action = baseline_agent(obs)
    return action


# -----------------------------
# GRADER
# -----------------------------
@app.post("/grader")
def grader(action: Action):
    import os

    # task from validator
    task = os.getenv("TASK_NAME")
    if task not in ["easy", "medium", "hard"]:
        task = "easy"

    # ❌ DO NOT use env.get_state()
    # ❌ DO NOT depend on env internal state

    email_text = action.response.lower()  # simulate input
    action_type = action.action

    correct = False

    if "urgent" in email_text and action_type == "escalate":
        correct = True
    elif "complaint" in email_text and action_type == "reply":
        correct = True
    elif "discount" in email_text and action_type == "ignore":
        correct = True
    elif "meeting" in email_text and action_type == "archive":
        correct = True

    if correct:
        if task == "easy":
            score = 0.8
        elif task == "medium":
            score = 0.85
        else:
            score = 0.9
    else:
        score = 0.2

    return {
        "score": float(score)
    }