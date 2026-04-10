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


@app.post("/grader")
def grader(action: Action):
    text = (action.response or "").lower().strip()
    action_type = (action.action or "").lower().strip()

    correct = False
    task = "easy"

    # ✅ HARD (check first to avoid overlap)
    if "discount" in text or "sale" in text:
        task = "hard"
        if action_type == "ignore":
            correct = True

    # ✅ MEDIUM
    elif "complaint" in text or "refund" in text:
        task = "medium"
        if action_type == "reply":
            correct = True

    elif "urgent" in text:
        task = "medium"
        if action_type == "escalate":
            correct = True

    # ✅ EASY
    elif "meeting" in text:
        task = "easy"
        if action_type == "archive":
            correct = True

    elif "invoice" in text:
        task = "easy"
        if action_type == "reply":
            correct = True

    # ✅ STRICT scoring
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