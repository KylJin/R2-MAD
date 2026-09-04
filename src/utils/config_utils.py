import os
import yaml


def load_configs_from_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_env_file(path: str = ".env") -> None:
    """
    Lightweight, zero-dependency loader for a `.env` file (KEY=VALUE lines).

    Reads simple `KEY=value` / `export KEY=value` lines, ignoring blanks and
    `#` comments and stripping surrounding quotes. Variables already present in
    the environment are NOT overwritten, so an explicit shell `export` still wins.
    Missing file is a no-op.
    """
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class LLMConfig:
    def __init__(self, general_config: dict, model_config: dict, name: str = ""):
        self.name = name
        
        # Serving backend: "vllm" (local GPU) or "openrouter" (remote API).
        self.backend = model_config.get("backend", general_config.get("backend", "vllm"))
        self.model = model_config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        self.model_path = model_config.get("model_path", None)
        self.max_token_length = model_config.get("max_token_length", general_config.get("max_token_length", 24576))
        self.temperature = model_config.get("temperature", general_config.get("temperature", 1))
        self.top_p = model_config.get("top_p", general_config.get("top_p", 1))
        self.max_tokens = model_config.get("max_tokens", general_config.get("max_tokens", 6144))

        # API backend (openrouter) settings; ignored by the vLLM backend.
        self.api_base = model_config.get("api_base", general_config.get("api_base", "https://openrouter.ai/api/v1"))
        self.api_key_env = model_config.get("api_key_env", general_config.get("api_key_env", "OPENROUTER_API_KEY"))
        self.max_workers = model_config.get("max_workers", general_config.get("max_workers", 12))
        self.max_retries = model_config.get("max_retries", general_config.get("max_retries", 5))
        self.request_timeout = model_config.get("request_timeout", general_config.get("request_timeout", 120))


class EmbeddingConfig:
    def __init__(self, model_config: dict):
        self.model = model_config.get("model", "bge-m3")
        self.model_path = model_config.get("model_path", None)
