import os
from env import EmailEnv
from baseline import baseline_agent
from openai import OpenAI

# ── Required environment variables ──────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Logging (spec-compliant) ─────────────────────────────
def log_start(task_id):
    # task= MUST match the task id in openenv.yaml exactly: easy | medium | hard
    print(f"[START] task={task_id} env=email-inbox-rl model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error="null"):
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error}",
        flush=True
    )

def log_end(success, steps, rewards):
    rs = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rs}", flush=True)

# ── LLM action picker ────────────────────────────────────
VALID_ACTIONS = {"archive", "reply", "escalate", "ignore"}

def pick_action(obs):
    try:
        prompt = (
            "You are an email triage assistant. Read the email below and "
            "reply with EXACTLY ONE word from: archive, reply, escalate, ignore\n\n"
            f"Email: {obs['email']}\n\n"
            "Rules:\n"
            "- spam / discount / sale / free / click -> ignore\n"
            "- urgent / server down / critical / security / breach -> escalate\n"
            "- complaint / refund / support / issue / wrong -> reply\n"
            "- meeting / invoice / normal -> archive\n\n"
            "One word answer:"
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
        )
        word = resp.choices[0].message.content.strip().lower().split()[0]
        if word in VALID_ACTIONS:
            return {"action": word, "response": obs["email"]}
    except Exception:
        pass
    return baseline_agent(obs)


# ── Run a single task episode ────────────────────────────
def run_task(task_id):
    """
    Runs one full episode for task_id.
    Emits: [START] task=<task_id> ... then [STEP]s ... then [END]
    task_id must exactly match an id in openenv.yaml: easy | medium | hard
    """
    env = EmailEnv()
    env.level = task_id
    obs = env.reset()

    rewards = []
    step = 0
    done = False
    action = {"action": "reply", "response": ""}

    log_start(task_id)  # task= matches openenv.yaml task id exactly

    while not done:
        error_str = "null"
        reward = 0.2  # safe default - strictly in (0,1)

        try:
            action = pick_action(obs)
            result = env.step(action)
            reward = float(result["reward"])
            reward = max(0.01, min(reward, 0.99))  # clamp - never 0.0 or 1.0
            done   = result["done"]
            obs    = result["observation"]
        except Exception as exc:
            reward = 0.2
            done   = True
            error_str = str(exc).replace("\n", " ")[:120]

        rewards.append(reward)
        step += 1
        log_step(step, action.get("action", "unknown"), reward, done, error_str)

    success = (sum(rewards) / len(rewards)) > 0.5 if rewards else False
    log_end(success, step, rewards)


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    # If validator sets OPENENV_TASK, run only that task.
    # Otherwise run all three - each gets its own [START]...[END] block.
    single_task = os.getenv("OPENENV_TASK", "").strip()

    if single_task in ("easy", "medium", "hard"):
        run_task(single_task)
    else:
        for task_id in ["easy", "medium", "hard"]:
            run_task(task_id)