def baseline_agent(obs):
    text = obs["email"].lower()

    scores = {
        "reply": 0,
        "escalate": 0,
        "ignore": 0,
        "archive": 0
    }

    # -------------------------
    # Smart keyword scoring
    # -------------------------

    # Customer issues → highest priority
    if any(w in text for w in ["refund", "complaint", "issue", "problem"]):
        scores["reply"] += 4

    # Urgent / critical
    if any(w in text for w in ["urgent", "server", "down", "critical"]):
        scores["escalate"] += 3

    # Spam / promotions
    if any(w in text for w in ["win", "free", "click", "discount", "sale"]):
        scores["ignore"] += 3

    # Normal work
    if any(w in text for w in ["meeting", "invoice"]):
        scores["archive"] += 2

    # -------------------------
    # Conflict resolution (KEY UPGRADE)
    # -------------------------
    if "urgent" in text and any(w in text for w in ["refund", "complaint"]):
        scores["reply"] += 3

    # -------------------------
    # Final decision
    # -------------------------
    best_action = max(scores, key=scores.get)

    if scores[best_action] == 0:
        best_action = "reply"

    return {
        "action": best_action,
        "response": "Taking appropriate action based on email context."
    }