import random

class EmailEnv:
    def __init__(self):
        self.level = "easy"
        self.reset()

    def reset(self):
        self.processed = 0
        self.mistakes = 0
        self.history = []

        self.current_email = self._generate_email()

        return self._get_obs()

    def _generate_email(self):
        samples = [
            {"text": "URGENT: Server is down!", "type": "priority"},
            {"text": "Customer complaint: refund needed", "type": "support"},
            {"text": "50% discount sale!!! click now", "type": "spam"},
            {"text": "Meeting at 5 PM", "type": "normal"},
            {"text": "Invoice issue, please check", "type": "support"},
        ]
        return random.choice(samples)

    def _get_obs(self):
        return {
            "email": self.current_email["text"],   # ✅ correct
            "processed": self.processed,
            "mistakes": self.mistakes,
            "history": self.history
        }

    def step(self, action):
        action_type = action.get("action", "")
        email_type = self.current_email["type"]

        correct = False

        # --- Decision logic ---
        if email_type == "priority" and action_type == "escalate":
            correct = True
        elif email_type == "support" and action_type == "reply":
            correct = True
        elif email_type == "spam" and action_type == "ignore":
            correct = True
        elif email_type == "normal" and action_type == "archive":
            correct = True

        # --- Reward ---
        if correct:
            reward = 1.0
        else:
            reward = 0.0   # ✅ FIXED (was -0.5, now valid range [0,1])
            self.mistakes += 1

        self.processed += 1

        self.history.append({
            "email": self.current_email["text"],
            "action": action_type,
            "correct": correct
        })

        done = self.processed >= 3

        self.current_email = self._generate_email()

        return {
            "observation": self._get_obs(),
            "reward": reward,
            "done": done,
            "info": {"expected": email_type}
        }

    def get_state(self):
        return {
            "email": self.current_email["text"],   # ✅ FINAL FIX (NO ERROR)
            "processed": self.processed,
            "mistakes": self.mistakes,
            "history": self.history
        }