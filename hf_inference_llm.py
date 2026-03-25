"""Hugging Face text generation for LangChain.

1. Tries cloud inference via ``urllib`` + hub task routing (same URLs as the official client).
2. If every cloud call fails (404, 401, timeouts, etc.), runs a **local** ``transformers``
   text-generation pipeline (default **distilgpt2**) so the chatbot still answers when the
   router is unavailable or the account has no serverless access.

Cloud path avoids ``InferenceClient`` because some ``huggingface_hub`` builds raise
``StopIteration`` on ``text_generation`` even with ``stream=False``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any

from huggingface_hub.inference._providers import get_provider_helper
from huggingface_hub.inference._providers._common import filter_none
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

_LOCAL_PIPE_LOCK = threading.Lock()
_LOCAL_PIPE: dict[str, Any] = {}


def _repetition_penalty_value() -> float | None:
    raw = os.getenv("HF_REPETITION_PENALTY", "1.18").strip()
    if not raw or raw.lower() in ("0", "none", "off"):
        return None
    try:
        v = float(raw)
    except ValueError:
        return 1.18
    return v if v > 1.0 else None


def _strip_echo_chat_pattern(text: str) -> str:
    """Truncate when the model starts another fake User/Answer turn."""
    for marker in ("\nUser:", "\nAnswer: User:", " Answer: User:"):
        idx = text.find(marker, 1)
        if idx != -1:
            return text[:idx].strip()
    return text


def _collapse_exact_repeated_prefixes(text: str, min_half: int = 24) -> str:
    """If output is `chunk + chunk + ...`, keep a single chunk (greedy LM loops)."""
    t = text.strip()
    while True:
        n = len(t)
        if n < min_half * 2:
            break
        found = False
        for half in range(n // 2, min_half - 1, -1):
            if t[:half] == t[half : 2 * half]:
                t = (t[:half] + t[2 * half :]).strip()
                found = True
                break
        if not found:
            break
    return t


def _collapse_consecutive_duplicate_sentences(text: str) -> str:
    """Drop back-to-back identical sentences (case-insensitive)."""
    s = text.strip()
    if not s:
        return s
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return s
    out = [parts[0]]
    for p in parts[1:]:
        if p.lower() == out[-1].lower():
            continue
        out.append(p)
    return " ".join(out).strip()


def _strip_leading_answer_label(text: str) -> str:
    return re.sub(r"(?is)^\s*(answer|reply)\s*:\s*", "", text).strip()


def _sanitize_completion(text: str) -> str:
    if not text:
        return text
    t = _strip_leading_answer_label(text)
    t = _strip_echo_chat_pattern(t)
    t = _collapse_exact_repeated_prefixes(t)
    t = _collapse_consecutive_duplicate_sentences(t)
    # Second pass: collapsing prefixes can expose new duplicate sentences
    t = _collapse_consecutive_duplicate_sentences(t)
    return t.strip()


def _dedupe_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _candidate_providers(
    model_id: str, provider_override: str | None, token: str
) -> list[str]:
    if provider_override:
        return [provider_override]
    try:
        from huggingface_hub.hf_api import HfApi

        info = HfApi(token=token).model_info(
            model_id, expand=["inferenceProviderMapping"]
        )
        pm = info.inference_provider_mapping
        if pm:
            return _dedupe_preserve([m.provider for m in pm])
    except Exception:
        pass
    return ["hf-inference"]


def _final_text(parsed: Any) -> str:
    if isinstance(parsed, dict) and "generated_text" in parsed:
        t = parsed["generated_text"]
        return t.strip() if isinstance(t, str) else str(t).strip()
    if isinstance(parsed, str):
        return parsed.strip()
    return str(parsed).strip()


def _urllib_generate(
    *,
    model_id: str,
    provider: str,
    prompt: str,
    parameters: dict[str, Any],
    token: str,
    timeout: float,
) -> str | None:
    try:
        helper = get_provider_helper(provider, "text-generation", model_id)
    except (StopIteration, ValueError):
        return None

    request_parameters = helper.prepare_request(
        inputs=prompt,
        parameters=parameters,
        extra_payload={"stream": False},
        headers={},
        model=model_id,
        api_key=token,
    )
    payload = request_parameters.json
    if payload is None:
        return None
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        request_parameters.url,
        data=body,
        headers={**request_parameters.headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("error"):
        return None
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]

    parsed = helper.get_response(data, request_parameters)
    out = _final_text(parsed)
    return out if out else None


def _truncate_for_local(prompt: str, max_chars: int = 6000) -> str:
    if len(prompt) <= max_chars:
        return prompt
    return (
        "[Earlier context omitted for local model context length.]\n\n"
        + prompt[-max_chars:]
    )


def _local_transformers_generate(prompt: str, max_new_tokens: int) -> str | None:
    if os.getenv("DISABLE_LOCAL_LLM_FALLBACK", "").lower() in ("1", "true", "yes"):
        return None
    repo = os.getenv("HF_LOCAL_LLM_REPO", "distilgpt2").strip() or "distilgpt2"
    try:
        from transformers import AutoTokenizer, pipeline
    except ImportError:
        return None

    prompt_in = _truncate_for_local(prompt)
    global _LOCAL_PIPE
    with _LOCAL_PIPE_LOCK:
        if repo not in _LOCAL_PIPE:
            tokenizer = AutoTokenizer.from_pretrained(repo)
            _LOCAL_PIPE[repo] = pipeline(
                "text-generation",
                model=repo,
                tokenizer=tokenizer,
            )
        pipe = _LOCAL_PIPE[repo]

    tok = pipe.tokenizer
    pad_id = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None) or 0
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
        "do_sample": False,
        "pad_token_id": pad_id,
    }
    rp = _repetition_penalty_value()
    if rp is not None:
        gen_kw["repetition_penalty"] = rp
    try:
        out = pipe(prompt_in, **gen_kw)
    except Exception:
        return None
    if not out:
        return None
    text = out[0].get("generated_text", "")
    if isinstance(text, str) and text.startswith(prompt_in):
        text = text[len(prompt_in) :].strip()
    return text.strip() if text.strip() else None


def _models_to_try(primary: str) -> list[str]:
    out = [primary]
    raw = os.getenv("HF_LLM_FALLBACK_IDS", "gpt2")
    for mid in [x.strip() for x in raw.split(",") if x.strip()]:
        if mid not in out:
            out.append(mid)
    return out


class HfInferenceTextGenLLM(LLM):
    """HF cloud inference via urllib; optional local transformers fallback."""

    model_id: str
    max_new_tokens: int = 384
    temperature: float = 0.3
    timeout: float = 180.0

    @property
    def _llm_type(self) -> str:
        return "hf_inference_text_generation"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        token = (
            os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or ""
        ).strip()

        provider_env = os.getenv("HF_INFERENCE_PROVIDER", "").strip()
        provider_override: str | None = provider_env or None
        if provider_override and provider_override.lower() in ("auto", "none"):
            provider_override = None

        param_dict: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": False,
            "return_full_text": False,
            "stop": stop if stop else None,
        }
        rp = _repetition_penalty_value()
        if rp is not None:
            param_dict["repetition_penalty"] = rp
        parameters = filter_none(param_dict)

        if token:
            for mid in _models_to_try(self.model_id):
                provs = _candidate_providers(mid, provider_override, token)
                for prov in provs:
                    text = _urllib_generate(
                        model_id=mid,
                        provider=prov,
                        prompt=prompt,
                        parameters=parameters,
                        token=token,
                        timeout=self.timeout,
                    )
                    if text:
                        return _sanitize_completion(text)

        local = _local_transformers_generate(
            prompt, max(32, min(self.max_new_tokens, 256))
        )
        if local:
            return _sanitize_completion(local)

        if not token:
            return (
                "Add **HF_TOKEN** to `.env` for cloud inference, or leave it unset to use "
                f"the local fallback model (**{os.getenv('HF_LOCAL_LLM_REPO', 'distilgpt2')}**). "
                "Install PyTorch + transformers (already pulled in by sentence-transformers)."
            )

        return (
            "Cloud Hugging Face inference did not return text (router/account/model), and the "
            f"local fallback also failed. Set **HF_LLM_REPO_ID** or **HF_LOCAL_LLM_REPO** "
            "(e.g. distilgpt2), ensure `transformers` works, or use **LLM_PROVIDER=ollama**. "
            "See https://huggingface.co/settings/inference-providers"
        )
