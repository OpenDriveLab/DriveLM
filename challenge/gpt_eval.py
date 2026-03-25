import os
import re
import numpy as np
import json
import argparse
from multiprocessing import Pool
from openai import OpenAI


# Supported LLM providers and their default configurations
PROVIDER_CONFIGS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,  # use OpenAI SDK default
        "default_model": "gpt-3.5-turbo",
    },
    "minimax": {
        "env_key": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.7",
    },
}


def _resolve_provider():
    """Auto-detect provider from environment variables.

    Priority: EVAL_LLM_PROVIDER env var > MINIMAX_API_KEY presence > OPENAI_API_KEY presence.
    """
    explicit = os.environ.get("EVAL_LLM_PROVIDER", "").lower()
    if explicit in PROVIDER_CONFIGS:
        return explicit

    if os.environ.get("MINIMAX_API_KEY"):
        return "minimax"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    return "openai"


def _strip_think_tags(content):
    """Strip <think>...</think> reasoning blocks from model output."""
    if content and "<think>" in content:
        return re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()
    return content


def _clamp_temperature(temperature, provider):
    """Clamp temperature to valid range for the provider."""
    if provider == "minimax" and temperature is not None:
        # MiniMax requires temperature in (0.0, 1.0]
        return max(0.01, min(temperature, 1.0))
    return temperature


class GPTEvaluation:
    """LLM-based evaluation scorer supporting multiple providers.

    Supported providers:
        - ``openai``: OpenAI API (default)
        - ``minimax``: MiniMax API (OpenAI-compatible)

    The provider is chosen by ``provider`` argument, the ``EVAL_LLM_PROVIDER``
    environment variable, or auto-detected from available API keys.
    """

    def __init__(self, provider=None, api_key=None, base_url=None, model=None):
        self.provider = provider or _resolve_provider()
        if self.provider not in PROVIDER_CONFIGS:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. "
                f"Choose from: {', '.join(PROVIDER_CONFIGS)}"
            )

        cfg = PROVIDER_CONFIGS[self.provider]
        resolved_key = api_key or os.environ.get(cfg["env_key"])
        if not resolved_key:
            raise ValueError(
                f"API key not found. Set {cfg['env_key']} environment variable "
                f"or pass api_key to GPTEvaluation()."
            )

        resolved_base_url = base_url or cfg["base_url"]
        self.default_model = model or cfg["default_model"]

        client_kwargs = {"api_key": resolved_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model=None):
        model = model or self.default_model
        temperature = _clamp_temperature(0.6, self.provider)
        response = self.client.chat.completions.create(
            model=model, messages=chatgpt_messages, temperature=temperature, max_tokens=max_tokens
        )
        reply = response.choices[0].message.content
        reply = _strip_think_tags(reply)
        total_tokens = response.usage.total_tokens
        return reply, total_tokens

    def prepare_chatgpt_message(self, prompt):
        system_message = "an evaluator who rates my answer based on the correct answer"
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": "{}".format(prompt)})

        return messages

    def forward(self, data):
        answer, GT = data
        prompts = "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, and you should be accurate to single digits like 62, 78, 41,etc. Output the number only"
        prompts = prompts + "This is the correct answer: " + GT + "This is my answer: " + answer

        output = ""
        messages = self.prepare_chatgpt_message(prompts)
        reply, total_tokens = self.call_chatgpt(messages, max_tokens=3000)

        output += reply
        output += "\n\n"

        output = output[:-2]

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-based Evaluation")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=list(PROVIDER_CONFIGS.keys()),
        help="LLM provider for evaluation (default: auto-detect from env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (default: provider-specific)",
    )
    args = parser.parse_args()

    data = [
        ("The ego vehicle should notice the bus next, as it is the third object in the image. The bus is stopped at the intersection, and the ego vehicle should be cautious when approaching the intersection to ensure it does not collide with the bus.", "Firstly, notice <c3,CAM_FRONT_LEFT,1075.5,382.8>. The object is a traffic sign, so the ego vehicle should continue at the same speed. Secondly, notice <c2,CAM_FRONT,836.3,398.3>. The object is a traffic sign, so the ego vehicle should accelerate and continue ahead. Thirdly, notice <c1,CAM_BACK,991.7,603.0>. The object is stationary, so the ego vehicle should continue ahead at the same speed."),
        # Add more data here
    ]

    evaluator = GPTEvaluation(provider=args.provider, model=args.model)
    print(f"Using provider: {evaluator.provider} (model: {evaluator.default_model})")

    with Pool(5) as p:  # Change the number based on your CPU cores
        scores = p.map(evaluator.forward, data)

    print(scores)
