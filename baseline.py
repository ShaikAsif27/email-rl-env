def baseline_agent(obs):
    text = obs["email"].lower()
    if "win" in text or "free" in text:
        act = "ignore"
    elif "urgent" in text or "login" in text or "meeting" in text:
        act = "escalate"
    elif "discount" in text:
        act = "archive"
    else:
        act = "reply"
    return {
        "action": act,
        "response": "Taking appropriate action based on email context."
    }
