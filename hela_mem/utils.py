"""Shared utilities for the HeLa-Mem LongMemEval release."""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Iterable

import numpy as np


_MODEL_CACHE: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()
_API_KEY_INDEX = 0
_API_KEY_LOCK = threading.Lock()
_FALLBACK_EMBEDDER = "HASH_EMBEDDER"

_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def env_bool(*names: str, default: bool = False) -> bool:
    value = _first_env(*names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(*names: str, default: int) -> int:
    value = _first_env(*names)
    return int(value) if value is not None else default


def env_float(*names: str, default: float) -> float:
    value = _first_env(*names)
    return float(value) if value is not None else default


def resolve_model_name() -> str:
    return _first_env("HELA_MEM_MODEL", "HEBBIAN_MODEL", default="gpt-4o-mini") or "gpt-4o-mini"


def resolve_base_url() -> str | None:
    return _first_env("OPENAI_BASE_URL", default=None)


def load_api_keys() -> list[str]:
    keys_value = _first_env("OPENAI_API_KEYS", default=None)
    if keys_value:
        keys = [key.strip() for key in re.split(r"[\n,]+", keys_value) if key.strip()]
        if keys:
            return keys

    keys_file = _first_env("OPENAI_API_KEYS_FILE", default=None)
    if keys_file and os.path.exists(keys_file):
        with open(keys_file, "r", encoding="utf-8") as handle:
            keys = [line.strip() for line in handle if line.strip()]
        if keys:
            return keys

    single_key = _first_env("OPENAI_API_KEY", default=None)
    return [single_key] if single_key else []


def has_llm_credentials() -> bool:
    return bool(load_api_keys())


def _get_next_api_key() -> str:
    keys = load_api_keys()
    if not keys:
        raise RuntimeError(
            "No OpenAI API key configured. Set OPENAI_API_KEY or OPENAI_API_KEYS before running LLM-backed steps."
        )

    global _API_KEY_INDEX
    with _API_KEY_LOCK:
        key = keys[_API_KEY_INDEX % len(keys)]
        _API_KEY_INDEX += 1
    return key


class OpenAIChatClient:
    """Small wrapper used by profile extraction helpers."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI

        if api_key is None:
            api_key = _get_next_api_key()
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


def create_chat_client(api_key: str | None = None) -> OpenAIChatClient:
    return OpenAIChatClient(api_key=api_key, base_url=resolve_base_url())


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def parse_timestamp(timestamp_str: str | None) -> datetime | None:
    if not timestamp_str:
        return None

    timestamp_str = str(timestamp_str).strip()
    patterns = (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d",
    )
    for pattern in patterns:
        try:
            return datetime.strptime(timestamp_str, pattern)
        except ValueError:
            continue
    return None


def normalize_vector(vec: Iterable[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr if norm == 0 else arr / norm


def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        bucket = hash(token) % dim
        vector[bucket] += 1.0

    return normalize_vector(vector)


def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    process_key = f"{os.getpid()}::{model_name}"

    if process_key not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if process_key not in _MODEL_CACHE:
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(model_name, device="cpu")
                    _MODEL_CACHE[process_key] = model
                except Exception as exc:
                    print(
                        f"Warning: failed to load sentence-transformers model '{model_name}'. "
                        f"Falling back to hashed embeddings. Error: {exc}"
                    )
                    _MODEL_CACHE[process_key] = _FALLBACK_EMBEDDER

    model = _MODEL_CACHE[process_key]
    if model == _FALLBACK_EMBEDDER:
        return _hash_embedding(text)

    with _MODEL_LOCK:
        embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    return embedding


def gpt_generate_answer(
    prompt: str,
    messages: list[dict[str, str]],
    client: OpenAIChatClient | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = 5,
) -> str:
    del prompt
    if model is None:
        model = resolve_model_name()

    if client is None:
        client = create_chat_client()

    for attempt in range(max_retries):
        try:
            return client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"LLM request failed after {max_retries} attempts: {exc}") from exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("Unreachable LLM retry path.")


def gpt_generate_answer_with_rotation(
    prompt: str,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = 5,
) -> str:
    del prompt
    if model is None:
        model = resolve_model_name()

    for attempt in range(max_retries):
        try:
            client = create_chat_client()
            return client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Rotating LLM request failed after {max_retries} attempts: {exc}") from exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("Unreachable LLM retry path.")


def heuristic_extract_keywords(text: str, limit: int = 3) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    counts: dict[str, int] = {}
    for token in tokens:
        if len(token) < 3 or token in _DEFAULT_STOPWORDS or token.isdigit():
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {token for token, _ in ranked[:limit]}


def llm_extract_keywords(text: str, client: OpenAIChatClient | None = None) -> set[str]:
    if not text.strip():
        return set()

    if not has_llm_credentials() and client is None:
        return heuristic_extract_keywords(text)

    prompt = (
        "Extract up to three topic keywords from the following text. "
        "Return only comma-separated keywords.\n\n"
        f"{text}"
    )
    messages = [
        {
            "role": "system",
            "content": "You extract concise topic keywords. Return only comma-separated keywords.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        output = gpt_generate_answer(prompt, messages, client=client, temperature=0.0, max_tokens=64)
        keywords = [keyword.strip() for keyword in output.split(",") if keyword.strip()]
        if keywords:
            return set(keywords[:3])
    except Exception:
        pass

    return heuristic_extract_keywords(text)


def compute_time_decay(timestamp_str: str, current_timestamp: str | None = None, tau: float | None = None) -> float:
    if tau is None:
        tau = env_float("HELA_MEM_TAU", "HEBBIAN_TAU", default=1e7)

    earlier = parse_timestamp(timestamp_str)
    if earlier is None:
        return 1.0

    later = parse_timestamp(current_timestamp) if current_timestamp else datetime.now()
    if later is None:
        later = datetime.now()

    delta = max((later - earlier).total_seconds(), 0.0)
    return float(np.exp(-delta / tau))
