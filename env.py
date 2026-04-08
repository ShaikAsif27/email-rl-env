import random
import os 
class EmailEnv:
    def __init__(self):
        self.level = os.getenv("TASK_NAME", "easy")  
        self.reset()

    def reset(self):
        self.processed = 0
        self.mistakes = 0
        self.history = []

        self.current_email = self._generate_email()

        return self._get_obs()

    # ✅ TASK DIFFERENTIATION (CRITICAL FIX)
    def _generate_email(self):
        if self.level == "easy":
            samples = [
                {"text": "Meeting at 5 PM", "type": "normal"},
                {"text": "Invoice issue, please check", "type": "support"},
            ]

        elif self.level == "medium":
            samples = [
                {"text": "Customer complaint: refund needed", "type": "support"},
                {"text": "URGENT: Server is down!", "type": "priority"},
            ]

        else:  # hard
            samples = [
                {"text": "50% discount sale!!! click now", "type": "spam"},
                {"text": "URGENT: Server is down!", "type": "priority"},
                {"text": "Customer complaint: refund needed", "type": "support"},
            ]

        return random.choice(samples)

    def _get_obs(self):
        return {
            "email": self.current_email["text"],
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

        # ✅ REWARD FIX (STRICT RANGE)
        if correct:
            if self.level == "easy":
                reward = 0.8
            elif self.level == "medium":
                reward = 0.85
            else:
                reward = 0.9
        else:
            reward = 0.2
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
            "email": self.current_email["text"],
            "processed": self.processed,
            "mistakes": self.mistakes,
            "history": self.history
        }