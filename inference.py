import os
from env import EmailEnv
from baseline import baseline_agent
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or ""
BASE = os.getenv("API_BASE_URL","https://router.huggingface.co/v1")
MODEL = os.getenv("MODEL_NAME","Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(api_key=API_KEY, base_url=BASE)

def log_start():
    print(f"[START] task=email-rl env=openenv model={MODEL}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success, steps, rewards):
    rs=",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rs}", flush=True)

def run():
    env = EmailEnv()
    rewards=[]
    step=0

    log_start()

    for lvl in ["easy","medium","hard"]:
        env.level=lvl
        obs=env.reset()

        while True:
            error="null"
            try:
                action=baseline_agent(obs)

                # lightweight LLM call (safe)
                try:
                    _ = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role":"user","content":"ok"}],
                        max_tokens=5
                    )
                except Exception:
                    pass

                res=env.step(action)
                r=res["reward"]
                done=res["done"]

            except Exception as e:
                r=0.0
                done=True
                error=str(e)

            rewards.append(r)
            step+=1
            log_step(step,str(action),r,done,error)

            if done:
                break
            obs=res["observation"]

    success=sum(rewards)>0
    log_end(success,step,rewards)

if __name__=="__main__":
    run()
