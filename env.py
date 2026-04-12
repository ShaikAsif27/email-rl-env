import os
import random


class EmailEnv:
    TASKS = ["easy", "medium", "hard"]

    # Email samples per level: (text, correct_action)
    EMAILS = {
        "easy": [
            {"text": "Meeting at 5 PM tomorrow", "correct": "archive"},
            {"text": "Invoice issue, please check payment", "correct": "reply"},
            {"text": "Project discussion scheduled", "correct": "archive"},
        ],
        "medium": [
            {"text": "Customer complaint: refund needed urgently", "correct": "reply"},
            {"text": "URGENT: Server is down! Fix immediately", "correct": "escalate"},
            {"text": "Client issue: product not working", "correct": "reply"},
        ],
        "hard": [
            {"text": "50% discount sale!!! Click now to buy", "correct": "ignore"},
            {"text": "URGENT: Critical security breach detected", "correct": "escalate"},
            {"text": "Customer complaint: wrong product delivered, refund needed", "correct": "reply"},
            {"text": "Limited time offer! Free coupons inside", "correct": "ignore"},
        ],
    }

    # Reward for a correct action — strictly in (0,1)
    CORRECT_REWARD = {"easy": 0.8, "medium": 0.85, "hard": 0.9}
    WRONG_REWARD = 0.2   # strictly > 0.0

    def __init__(self):
        task = os.getenv("OPENENV_TASK", "easy")
        self.level = task if task in self.TASKS else "easy"
        self._reset_state()

    def _reset_state(self):
        self.processed = 0
        self.mistakes = 0
        self.history = []
        self.current_email = random.choice(self.EMAILS[self.level])

    def reset(self):
        task = os.getenv("OPENENV_TASK", self.level)
        if task in self.TASKS:
            self.level = task
        self._reset_state()
        return self._obs()

    def _obs(self):
        return {
            "email": self.current_email["text"],
            "level": self.level,
            "processed": self.processed,
            "mistakes": self.mistakes,
            "history": self.history,
        }

    def step(self, action: dict):
        # Episode already done guard
        if self.processed >= 3:
            return {"observation": self._obs(), "reward": self.WRONG_REWARD,
                    "done": True, "info": {"message": "episode finished"}}

        action_type = (action.get("action") or "").lower().strip()
        correct = (action_type == self.current_email["correct"])

        email = self.current_email["text"].lower()
        if correct:
            reward = self.CORRECT_REWARD[self.level]
            if "urgent" in email and action_type == "escalate":
                reward += 0.05
            if "complaint" in email and action_type == "reply":
                reward += 0.05
        else:
            reward = self.WRONG_REWARD
            self.mistakes += 1
            # Penalty for bad critical decisions
            if "urgent" in email and action_type == "ignore":
                reward -= 0.1
        
        # Clamp reward STRICTLY (VERY IMPORTANT)
        reward = max(0.01, min(0.99, reward))


        self.history.append({
            "email": self.current_email["text"],
            "action": action_type,
            "correct": correct,
        })
        self.processed += 1
        done = self.processed >= 3

        if not done:
            self.current_email = random.choice(self.EMAILS[self.level])

        return {
            "observation": self._obs(),
            "reward": float(reward),
            "done": done,
            "info": {"expected": self.current_email["correct"]},
        }

    def get_state(self):
        return {
            "email": self.current_email["text"],
            "level": self.level,
            "processed": self.processed,
            "mistakes": self.mistakes,
            "history": self.history,
        }
