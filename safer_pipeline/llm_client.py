"""
llm_client.py

Provider-agnostic LLM call wrapper for the SAFER 3-part pipeline
(Task Planning LLM, Safety Planning LLM -- and room for a 3rd model if the
Safety-Weighted Tree-of-Thought branch-scorer ever needs to be split out
into its own call).

Adapted from COHERENT's PEFA/LLM.py `lm_engine` pattern (see
coherent_upstream/COHERENT/src/experiment/PEFA/LLM.py), but:
  - not tied to OpenAI specifically -- each LLMClient instance can point at
    a different provider/model, since the plan is to use different LLM
    APIs for different roles in the pipeline
  - no OmniGibson / scene-graph specific plumbing
  - API keys are ALWAYS read from environment variables, never hardcoded
    (see the ai_scene_scan.py key-leak fix from earlier -- same rule
    applies here)

Every call site (task_planner.py, safety_checker.py, ...) constructs its
own LLMClient with its own model name, so swapping which model handles
which pipeline role is a one-line change, not a rewrite.
"""

import os
import json
import time


class LLMClient:
    """Thin wrapper around a single LLM endpoint.

    PLACEHOLDER IMPLEMENTATION -- `_call_openai` is wired up as a real
    example since that's what COHERENT used, but `provider` can be swapped
    for any other backend (Anthropic, local vLLM server, OpenRouter, etc.)
    by adding another `_call_<provider>` method and branching in `generate`.
    Every provider branch must return a plain string.
    """

    def __init__(self, model: str, provider: str = "openai",
                 api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.2, max_tokens: int = 512,
                 role: str = "unnamed"):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.role = role  # e.g. "task_planner", "safety_checker" -- just for logging

        self.api_key = os.environ.get(api_key_env)
        if self.api_key is None:
            # Don't crash at import time -- a lot of scaffolding/testing
            # happens before real keys exist. Just warn loudly so a silent
            # placeholder call doesn't get mistaken for a real one.
            print(f"[llm_client] WARNING: env var {api_key_env} not set. "
                  f"{role} LLMClient will use the placeholder generator.")

    def generate(self, messages: list[dict]) -> str:
        """messages: standard chat format, e.g.
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]

        Returns the raw text of the model's reply. Callers are responsible
        for parsing structured content (JSON, action labels, etc.) out of it.
        """
        if self.api_key is None:
            return self._placeholder_generate(messages)

        if self.provider == "openai":
            return self._call_openai(messages)

        if self.provider == "openrouter":
            return self._call_openrouter(messages)

        raise ValueError(f"Unknown provider '{self.provider}' -- add a "
                          f"_call_{self.provider} method to LLMClient.")

    # ------------------------------------------------------------------
    # Real provider implementations
    # ------------------------------------------------------------------

    def _call_openai(self, messages: list[dict]) -> str:
        from openai import OpenAI, OpenAIError
        import backoff

        client = OpenAI(api_key=self.api_key)

        @backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)
        def _do_call():
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        return _do_call()

    def _call_openrouter(self, messages: list[dict]) -> str:
        """OpenRouter exposes most non-OpenAI models (Qwen, Kimi,
        DeepSeek, Nemotron, Claude, ...) behind one OpenAI-compatible
        endpoint -- so extension #6 (expanded model comparisons) mostly
        just needs different `model` strings pointed at this same method,
        not a separate integration per provider.
        REMEMBER: revoke/regenerate any key that was ever pasted as a
        literal string anywhere -- always read from the environment.
        """
        from openai import OpenAI, OpenAIError
        import backoff

        client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

        @backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)
        def _do_call():
            response = client.chat.completions.create(
                model=self.model,  # e.g. "qwen/qwen-2.5-72b-instruct", "moonshotai/kimi-k2"
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        return _do_call()

    # ------------------------------------------------------------------
    # Placeholder, used until real API keys are wired in per role
    # ------------------------------------------------------------------

    def _placeholder_generate(self, messages: list[dict]) -> str:
        """Deterministic stand-in so the rest of the pipeline (arbitration,
        CBF gating, logging, dialogue history) can be built and tested
        end-to-end before real model access exists for every role.

        Returns a minimal, obviously-fake response shaped like what the
        real prompts ask for, tagged so it's unmistakable in logs.
        """
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        )
        return json.dumps({
            "_placeholder": True,
            "role": self.role,
            "model": self.model,
            "note": "No API key set -- replace via api_key_env before real runs.",
            "echo_prompt_len": len(last_user_msg),
        })


def load_prompt(path: str, **substitutions) -> str:
    """Reads a prompt template file and substitutes #TOKEN# placeholders,
    matching COHERENT's `.replace('#OBSERVATION#', ...)` style so the
    prompt-template files stay easy to hand-edit without touching code.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for key, value in substitutions.items():
        text = text.replace(f"#{key.upper()}#", str(value))
    return text
