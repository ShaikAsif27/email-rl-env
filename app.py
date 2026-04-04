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
    return obs["observation"] if "observation" in obs else obss


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
    result = env.step({
        "action": action.action,
        "response": action.response
    })

    score = result.get("reward", 0.0)

    return {
        "score": float(score),
        "done": result.get("done", False),
        "info": result.get("info", {})
    }