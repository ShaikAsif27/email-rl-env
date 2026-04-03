def get_task_emails(level):
    if level == "easy":
        return [
            {"text":"Meeting at 5 PM","sender":"boss","label":"important"},
            {"text":"50% discount sale","sender":"store","label":"promotion"},
        ]
    elif level == "medium":
        return [
            {"text":"Submit report urgently","sender":"manager","label":"important"},
            {"text":"Win free iPhone","sender":"spam","label":"spam"},
        ]
    else:
        return [
            {"text":"Catch up this weekend","sender":"friend","label":"social"},
            {"text":"Unauthorized login alert","sender":"bank","label":"important"},
        ]
