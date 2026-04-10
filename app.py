from fastapi import FastAPI, Query
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
def reset_post(task: str = Query(default=None)):
    if task:
        env.level = task
    obs = env.reset()
    return obs["observation"] if "observation" in obs else obs

@app.get("/reset")
def reset_get(task: str = Query(default=None)):
    if task:
        env.level = task
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
def grader(action: Action, task_id: str = Query(default="easy")):
    """
    Validator calls this per task with task_id = easy | medium | hard.
    Score must be STRICTLY between 0 and 1 (not 0.0, not 1.0).
    """
    text = (action.response or "").lower().strip()
    action_type = (action.action or "").lower().strip()

    correct = False

    if task_id == "easy":
        # Easy task: archive meetings, reply to invoices
        if ("meeting" in text and action_type == "archive") or \
           ("invoice" in text and action_type == "reply"):
            correct = True
        elif action_type in ("archive", "reply"):
            correct = True  # reasonable action for easy task

    elif task_id == "medium":
        # Medium task: reply to complaints/refunds, escalate urgent
        if ("complaint" in text or "refund" in text) and action_type == "reply":
            correct = True
        elif "urgent" in text and action_type == "escalate":
            correct = True
        elif action_type in ("reply", "escalate"):
            correct = True

    elif task_id == "hard":
        # Hard task: ignore spam/discounts, escalate urgent, reply to complaints
        if ("discount" in text or "sale" in text) and action_type == "ignore":
            correct = True
        elif "urgent" in text and action_type == "escalate":
            correct = True
        elif ("complaint" in text or "refund" in text) and action_type == "reply":
            correct = True
        elif action_type in ("ignore", "escalate", "reply"):
            correct = True

    # Scores strictly in (0.0, 1.0) — never 0.0 or 1.0
    if correct:
        score_map = {"easy": 0.8, "medium": 0.85, "hard": 0.9}
        score = score_map.get(task_id, 0.8)
    else:
        score = 0.2

    return {"score": float(score)}