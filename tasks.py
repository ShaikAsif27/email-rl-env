def get_task_emails(level):
    if level == "easy":
        return [
            {"text": "Meeting at 5 PM", "sender": "boss", "label": "important"},
            {"text": "Invoice generated", "sender": "finance", "label": "important"},
        ]
    elif level == "medium":
        return [
            {"text": "Customer complaint refund request", "sender": "client", "label": "important"},
            {"text": "URGENT: server down", "sender": "ops", "label": "critical"},
        ]
    else:
        return [
            {"text": "50% discount sale", "sender": "store", "label": "spam"},
            {"text": "Critical security alert", "sender": "bank", "label": "important"},
        ]