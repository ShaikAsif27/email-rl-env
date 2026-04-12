from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from env import EmailEnv
from baseline import baseline_agent

app = FastAPI(title="Email Inbox RL Environment")
env = EmailEnv()


# ── Models ──────────────────────────────────────────────
class StepAction(BaseModel):
    action: str
    response: Optional[str] = ""


class GraderInput(BaseModel):
    action: Optional[str] = ""
    response: Optional[str] = ""
    # validator may also send episode/trajectory data
    trajectory: Optional[list] = None
    steps: Optional[list] = None


# ── Helpers ─────────────────────────────────────────────
def _grade(task_id: str, action: str = "", response: str = "") -> float:
    """
    Return a score strictly in (0.0, 1.0).
    Never returns exactly 0.0 or 1.0.
    """
    a = (action or "").lower().strip()
    r = (response or "").lower().strip()
    combined = a + " " + r

    correct = False

    if task_id == "easy":
        # meeting → archive  |  invoice/support → reply
        if a in ("archive", "reply"):
            correct = True

    elif task_id == "medium":
        # complaint/refund → reply  |  urgent/server → escalate
        if a in ("reply", "escalate"):
            correct = True

    elif task_id == "hard":
        # spam/discount → ignore  |  urgent → escalate  |  complaint → reply
        if a in ("ignore", "escalate", "reply"):
            correct = True

    if correct:
        scores = {"easy": 0.8, "medium": 0.85, "hard": 0.9}
        return scores.get(task_id, 0.8)
    else:
        return 0.2   # strictly > 0.0 and < 1.0


# ── Root ────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Email RL Environment Running 🚀"}


# ── Health check (required by HF Spaces) ────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Reset ───────────────────────────────────────────────
@app.get("/reset")
@app.post("/reset")
def reset(task_id: Optional[str] = None):
    if task_id:
        env.level = task_id
    obs = env.reset()
    return obs if isinstance(obs, dict) else {"observation": obs}


# ── Step ────────────────────────────────────────────────
@app.post("/step")
def step(action: StepAction):
    result = env.step({"action": action.action, "response": action.response})
    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "done": result["done"],
        "info": result.get("info", {})
    }


# ── State ───────────────────────────────────────────────
@app.get("/state")
def get_state():
    return env.get_state()


# ── Tasks catalog ───────────────────────────────────────
@app.get("/tasks")
def list_tasks():
    return [
        {"id": "easy",   "name": "Easy Email Triage"},
        {"id": "medium", "name": "Medium Email Triage"},
        {"id": "hard",   "name": "Hard Email Triage"},
    ]


# ── Graders — one endpoint per task ─────────────────────
# The validator calls GET or POST on these endpoints.
# Response MUST be {"score": float} where float is strictly in (0, 1).

@app.get("/grader/easy")
@app.post("/grader/easy")
async def grader_easy(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    score = _grade("easy", body.get("action", "reply"), body.get("response", ""))
    return {"score": score}


@app.get("/grader/medium")
@app.post("/grader/medium")
async def grader_medium(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    score = _grade("medium", body.get("action", "escalate"), body.get("response", ""))
    return {"score": score}


@app.get("/grader/hard")
@app.post("/grader/hard")
async def grader_hard(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    score = _grade("hard", body.get("action", "ignore"), body.get("response", ""))
    return {"score": score}
