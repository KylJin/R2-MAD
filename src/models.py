import os
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Union, Optional, Tuple

from src.utils import LLMConfig, EmbeddingConfig, TokenUsageTracker


def build_language_model(llm_config: LLMConfig):
    """Return the language model implementation for the configured backend."""
    if llm_config.backend == "openrouter":
        return OpenRouterLanguageModel(llm_config)
    return LanguageModel(llm_config)


class LanguageModel:
    """Wraps a vLLM model; accepts message lists and returns structured results."""
    def __init__(self, llm_config: LLMConfig):
        import torch
        from vllm import LLM, SamplingParams

        self.llm_config = llm_config
        self.model_name = llm_config.name or llm_config.model.split("/")[-1]

        llm_name_path = (
            self.llm_config.model_path
            if self.llm_config.model_path not in (None, "None")
            else self.llm_config.model
        )
        print(f">>>>>> LLM name path: {llm_name_path}")

        self.llm = LLM(
            model=llm_name_path,
            tensor_parallel_size=getattr(self.llm_config, "tensor_parallel_size", 1),
            dtype=torch.bfloat16,
            max_model_len=getattr(self.llm_config, "max_token_length", 24576),
            gpu_memory_utilization=0.85,
        )
        self.sampling_params = SamplingParams(
            temperature=getattr(self.llm_config, "temperature", 1),
            top_p=getattr(self.llm_config, "top_p", 1),
            max_tokens=getattr(self.llm_config, "max_tokens", 6144),
            stop_token_ids=[self.llm.get_tokenizer().eos_token_id],
        )

        self.token_usage_tracker = TokenUsageTracker()
    
    def __call__(self, batch_messages: List[List[Dict[str, str]]], task_name: str = "default") -> Dict[str, Any]:
        """
        Args:
            batch_messages: list of message lists, each message list is
                            [{"role": ..., "content": ...}, ...]
            task_name: label used to group token usage statistics

        Returns dict with keys:
            results, cur_batch_input_tokens, cur_batch_output_tokens
        """
        tokenizer = self.llm.get_tokenizer()
        tokenized_texts = tokenizer.apply_chat_template(
            batch_messages, tokenize=False, add_generation_prompt=True
        )

        # Handle over-long queries
        max_len = self.llm.llm_engine.model_config.max_model_len
        valid_indices, valid_texts = [], []
        for i, text in enumerate(tokenized_texts):
            token_len = len(tokenizer.encode(text))
            if token_len > max_len:
                print(
                    f"[WARNING] [{task_name}] Skipping request {i}: "
                    f"prompt length {token_len} > max_model_len {max_len}"
                )
            else:
                valid_indices.append(i)
                valid_texts.append(text)
        if not valid_texts:
            return {"responses": [""] * len(batch_messages), "cur_batch_input_tokens": 0, "cur_batch_output_tokens": 0}

        outputs = self.llm.generate(
            valid_texts,
            sampling_params=self.sampling_params
        )

        valid_responses, total_in, total_out = self._process_outputs(outputs)

        responses_list = [""] * len(batch_messages)
        for idx, resp in zip(valid_indices, valid_responses):
            responses_list[idx] = resp

        self.token_usage_tracker.update_usage(
            self.llm_config.model, total_in, total_out, task_name=task_name
        )
        print(
            f">>>>>> Token usage [{task_name}]: input={total_in}, output={total_out}, "
            f"total={total_in + total_out}"
        )

        return {
            "responses": responses_list,
            "cur_batch_input_tokens": total_in,
            "cur_batch_output_tokens": total_out,
        }

    def _process_outputs(self, outputs):
        responses_list = []
        total_in = total_out = 0

        for output in outputs:
            response = output.outputs[0].text
            responses_list.append(response)

            total_in += len(output.prompt_token_ids)
            total_out += len(output.outputs[0].token_ids)

        return responses_list, total_in, total_out

    def get_token_usage_summary(self):
        return self.token_usage_tracker.get_summary()


class EmbeddingModel:
    """Wraps a SentenceTransformer model for text embedding."""
    def __init__(self, embedding_config: EmbeddingConfig, device: str = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.embedding_config = embedding_config

        model_name_path = (
            self.embedding_config.model_path
            if self.embedding_config.model_path not in (None, "None")
            else self.embedding_config.model
        )
        # An explicit device (e.g. "cuda:2" from --embed_gpu_id) wins
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">>>>>> Loading embedding model '{model_name_path}' on {device}")

        self.model = SentenceTransformer(
            model_name_path,
            device=device,
            trust_remote_code=True,
        )

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True,
               show_progress_bar: bool = False) -> Union[List[float], List[List[float]]]:
        try:
            is_single = isinstance(texts, str)
            inputs = [texts] if is_single else texts
            
            embeddings = self.model.encode(
                inputs, 
                normalize_embeddings=normalize_embeddings, 
                show_progress_bar=show_progress_bar
            )

            return embeddings[0].tolist() if is_single else [e.tolist() for e in embeddings]
        
        except Exception as e:
            raise Exception(f"Failed to encode texts: {str(e)}")


class OpenRouterLanguageModel:
    """
    OpenRouter (OpenAI-compatible) backend that mirrors the interface of ``LanguageModel``:

        model(batch_messages, task_name) -> {"responses", "cur_batch_input_tokens",
                                             "cur_batch_output_tokens"}
        model.model_name
        model.get_token_usage_summary()
    
    A batch is fanned out over a thread pool (each request is a blocking, IO-bound
    HTTP call). Concurrency is capped by ``max_workers`` and every request retries
    on rate-limit / transient errors with exponential backoff + jitter, honoring the
    server's ``Retry-After`` when present. A request that still fails yields "" so a
    single bad call never aborts the whole debate round (same fallback as the vLLM
    backend for over-long / empty prompts).
    """

    def __init__(self, llm_config: LLMConfig):
        from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError

        self.llm_config = llm_config
        self.model_name = llm_config.name or llm_config.model.split("/")[-1]
        self.model_id = llm_config.model

        # Exception types stored on the instance so the lazy import stays local.
        self._api_error = APIError
        self._retryable = (RateLimitError, APITimeoutError, APIConnectionError)

        api_key = os.environ.get(llm_config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"OpenRouter backend requires the '{llm_config.api_key_env}' environment "
                f"variable to be set for model '{self.model_name}'."
            )

        # max_retries=0: we own retry/backoff so the SDK's built-in retries don't
        # multiply the request count on top of ours.
        self.client = OpenAI(
            base_url=llm_config.api_base,
            api_key=api_key,
            timeout=llm_config.request_timeout,
            max_retries=0,
        )

        self.max_workers = max(1, int(llm_config.max_workers))
        self.max_retries = max(0, int(llm_config.max_retries))
        self.temperature = llm_config.temperature
        self.top_p = llm_config.top_p
        self.max_tokens = llm_config.max_tokens

        self.token_usage_tracker = TokenUsageTracker()

        print(
            f">>>>>> OpenRouter model: {self.model_id} "
            f"(max_workers={self.max_workers}, max_retries={self.max_retries})"
        )

    def __call__(self, batch_messages: List[List[Dict[str, str]]], task_name: str = "default") -> Dict[str, Any]:
        if not batch_messages:
            return {"responses": [], "cur_batch_input_tokens": 0, "cur_batch_output_tokens": 0}

        from concurrent.futures import as_completed
        from tqdm import tqdm

        # Submit all requests, then consume them as they finish so tqdm reflects real progress. 
        # Results are placed back by original index to keep output order.
        results: List[Optional[Tuple[str, int, int]]] = [None] * len(batch_messages)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._complete_one_query, messages, task_name): i
                for i, messages in enumerate(batch_messages)
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(batch_messages),
                desc=f"[{task_name}] {self.model_name}",
                leave=False,
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[WARNING] [{task_name}] request worker failed at index {idx}: {e}")
                    results[idx] = ("", 0, 0)

        responses_list: List[str] = []
        total_in = total_out = 0
        for text, in_tok, out_tok in results:
            responses_list.append(text)
            total_in += in_tok
            total_out += out_tok

        self.token_usage_tracker.update_usage(
            self.model_id, total_in, total_out, task_name=task_name
        )
        print(
            f">>>>>> Token usage [{task_name}]: input={total_in}, output={total_out}, "
            f"total={total_in + total_out}"
        )

        return {
            "responses": responses_list,
            "cur_batch_input_tokens": total_in,
            "cur_batch_output_tokens": total_out,
        }

    def _complete_one_query(self, messages: List[Dict[str, str]], task_name: str) -> Tuple[str, int, int]:
        """Single chat completion with retry/backoff. Returns (text, in_tokens, out_tokens)."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", 0) or 0
                out_tok = getattr(usage, "completion_tokens", 0) or 0
                return text, in_tok, out_tok
            except self._retryable as e:
                if attempt >= self.max_retries:
                    print(f"[WARNING] [{task_name}] request failed after {attempt + 1} attempts: {e}")
                    return "", 0, 0
                time.sleep(self._backoff_seconds(attempt, e))
            except json.JSONDecodeError as e:
                # Gateway occasionally returns HTML/empty body under heavy load.
                if attempt >= self.max_retries:
                    print(
                        f"[WARNING] [{task_name}] malformed JSON response after "
                        f"{attempt + 1} attempts: {e}"
                    )
                    return "", 0, 0
                time.sleep(self._backoff_seconds(attempt, e))
            except self._api_error as e:
                # Non-retryable API error (bad request, auth, etc.): give up on this one.
                print(f"[WARNING] [{task_name}] non-retryable API error: {e}")
                return "", 0, 0
            except Exception as e:
                if attempt >= self.max_retries:
                    print(
                        f"[WARNING] [{task_name}] unexpected error after "
                        f"{attempt + 1} attempts: {type(e).__name__}: {e}"
                    )
                    return "", 0, 0
                time.sleep(self._backoff_seconds(attempt, e))

        return "", 0, 0

    def _backoff_seconds(self, attempt: int, error: Exception) -> float:
        """Honor Retry-After when the server sends it; otherwise exponential backoff + jitter."""
        retry_after = self._parse_retry_after(error)
        if retry_after is not None:
            return retry_after + random.random()
        return min(2.0 ** attempt, 30.0) + random.random()

    @staticmethod
    def _parse_retry_after(error: Exception) -> Optional[float]:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_token_usage_summary(self):
        return self.token_usage_tracker.get_summary()
