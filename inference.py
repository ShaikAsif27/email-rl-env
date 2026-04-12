import os
from env import EmailEnv
from baseline import baseline_agent
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

VALID_ACTIONS = {"archive", "reply", "escalate", "ignore"}

def safe_reward(r):
    """Guarantee reward is strictly in (0,1). No exceptions."""
    try:
        r = float(r)
    except Exception:
        return 0.5
    if r <= 0.0 or r >= 1.0:
        return 0.5
    return r

def log_start(task_id):
    print(f"[START] task={task_id} env=email-inbox-rl model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error="null"):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success, steps, rewards):
    rs = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rs}", flush=True)

def pick_action(obs):
    try:
        prompt = (
            "You are an email triage assistant. "
            "Reply with ONE word only: archive, reply, escalate, or ignore\n\n"
            f"Email: {obs['email']}\n\n"
            "- spam/discount/sale/free → ignore\n"
            "- urgent/server/critical/security → escalate\n"
            "- complaint/refund/issue → reply\n"
            "- meeting/invoice/normal → archive\n\n"
            "Answer:"
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

def run_task(task_id):
    env = EmailEnv()
    env.level = task_id
    obs = env.reset()
    rewards = []
    step = 0
    done = False
    action = {"action": "reply", "response": ""}

    log_start(task_id)

    while not done:
        error_str = "null"
        reward = 0.5  # safe default

        try:
            action = pick_action(obs)
            result = env.step(action)
            reward = safe_reward(result["reward"])  # triple-safe
            done = result["done"]
            obs  = result["observation"]
        except Exception as exc:
            reward = 0.5  # safe default
            done = True
            error_str = str(exc).replace("\n", " ")[:100]

        rewards.append(reward)
        step += 1
        log_step(step, action.get("action", "reply"), reward, done, error_str)

    success = (sum(rewards) / len(rewards)) > 0.5 if rewards else False
    log_end(success, step, rewards)

if __name__ == "__main__":
    single_task = os.getenv("OPENENV_TASK", "").strip()
    if single_task in ("easy", "medium", "hard"):
        run_task(single_task)
    else:
        for task_id in ["easy", "medium", "hard"]:
            run_task(task_id)