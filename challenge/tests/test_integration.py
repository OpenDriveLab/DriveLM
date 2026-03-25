"""Integration tests for GPTEvaluation with live LLM APIs.

These tests are skipped unless the corresponding API key is set.
Run with: pytest challenge/tests/test_integration.py -v
"""

import os
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpt_eval import GPTEvaluation

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SAMPLE_DATA = (
    "The ego vehicle should slow down because there is a pedestrian crossing.",
    "The ego vehicle should decelerate and yield to the pedestrian on the crosswalk.",
)


@pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")
class TestMiniMaxIntegration:
    """Live integration tests against MiniMax API."""

    def test_basic_evaluation(self):
        evaluator = GPTEvaluation(provider="minimax")
        result = evaluator.forward(SAMPLE_DATA)
        score = float(result.strip())
        assert 0 <= score <= 100, f"Score out of range: {score}"

    def test_call_chatgpt_returns_content(self):
        evaluator = GPTEvaluation(provider="minimax")
        messages = [{"role": "user", "content": "Reply with the number 42 only."}]
        reply, tokens = evaluator.call_chatgpt(messages, max_tokens=200)
        assert reply is not None
        assert len(reply) > 0
        assert tokens > 0

    def test_highspeed_model(self):
        evaluator = GPTEvaluation(provider="minimax", model="MiniMax-M2.7-highspeed")
        result = evaluator.forward(SAMPLE_DATA)
        score = float(result.strip())
        assert 0 <= score <= 100


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
class TestOpenAIIntegration:
    """Live integration tests against OpenAI API."""

    def test_basic_evaluation(self):
        evaluator = GPTEvaluation(provider="openai")
        result = evaluator.forward(SAMPLE_DATA)
        score = float(result.strip())
        assert 0 <= score <= 100, f"Score out of range: {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
