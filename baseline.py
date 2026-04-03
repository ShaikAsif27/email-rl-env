def baseline_agent(obs):
    text = obs["email"].lower()

    # Spam detection
    if "win" in text or "free" in text or "click" in text:
        act = "ignore"

    # Urgent / critical issues
    elif "urgent" in text or "server" in text or "down" in text:
        act = "escalate"

    # Customer support cases
    elif "refund" in text or "complaint" in text or "issue" in text:
        act = "reply"

    # Promotions
    elif "discount" in text or "sale" in text:
        act = "archive"

    # Default safe action
    else:
        act = "reply"

    return {
        "action": act,
        "response": "Taking appropriate action based on email context."
    }