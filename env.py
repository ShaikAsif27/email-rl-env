from tasks import get_task_emails

class EmailEnv:
    def __init__(self):
        self.level = "easy"
        self.reset()

    def reset(self):
        self.inbox = get_task_emails(self.level)
        self.index = 0
        self.mistakes = 0
        self.total_reward = 0
        self.history = []
        return self._obs()

    def _obs(self):
        if self.index >= len(self.inbox):
            return None
        e = self.inbox[self.index]
        return {
            "email": e["text"],
            "sender": e["sender"],
            "processed": self.index,
            "mistakes": self.mistakes,
            "history": self.history[-2:]
        }

    def step(self, action):
        e = self.inbox[self.index]
        expected = e["label"]
        reward = 0.0

        # decision logic
        if action["action"] == "ignore" and expected == "spam":
            reward += 0.6
        elif action["action"] == "reply" and expected in ["important","social"]:
            reward += 0.6
        elif action["action"] == "archive" and expected == "promotion":
            reward += 0.5
        elif action["action"] == "escalate" and expected == "important":
            reward += 0.7
        else:
            reward -= 0.4
            self.mistakes += 1

        # response quality
        if len(action.get("response","").split()) > 5:
            reward += 0.2

        # trajectory penalty
        if self.mistakes >= 2:
            reward -= 0.2

        self.total_reward += reward

        self.history.append({
            "email": e["text"],
            "action": action["action"],
            "reward": round(reward,2)
        })

        self.index += 1
        done = self.index >= len(self.inbox)

        return {
            "observation": self._obs(),
            "reward": round(reward,2),
            "done": done,
            "info": {"expected": expected}
        }

    def state(self):
        return {
            "processed": self.index,
            "mistakes": self.mistakes,
            "total_reward": round(self.total_reward,2)
        }
