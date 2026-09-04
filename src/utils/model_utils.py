class TokenUsageTracker:
    """Tracks token usage and calculates costs, optionally broken down by task."""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self.usage_history = []
        self.task_usage = {}  # task_name -> {"input_tokens": int, "output_tokens": int}

    def update_usage(self, model: str, input_tokens: int, output_tokens: int, task_name: str = "default"):
        usage_record = {
            "model": model,
            "task": task_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.usage_history.append(usage_record)

        if task_name not in self.task_usage:
            self.task_usage[task_name] = {"input_tokens": 0, "output_tokens": 0}
        self.task_usage[task_name]["input_tokens"] += input_tokens
        self.task_usage[task_name]["output_tokens"] += output_tokens

    def get_summary(self):
        by_task = {
            task: {
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "total_tokens": stats["input_tokens"] + stats["output_tokens"],
            }
            for task, stats in self.task_usage.items()
        }
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": len(self.usage_history),
            "by_task": by_task,
            "history": self.usage_history,
        }
