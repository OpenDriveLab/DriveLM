"""Unit tests for GPTEvaluation multi-provider support."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpt_eval import GPTEvaluation, PROVIDER_CONFIGS, _resolve_provider, _clamp_temperature, _strip_think_tags


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


class TestResolveProvider:
    """Tests for _resolve_provider auto-detection logic."""

    def test_explicit_env_openai(self):
        with patch.dict(os.environ, {"EVAL_LLM_PROVIDER": "openai"}, clear=False):
            assert _resolve_provider() == "openai"

    def test_explicit_env_minimax(self):
        with patch.dict(os.environ, {"EVAL_LLM_PROVIDER": "minimax"}, clear=False):
            assert _resolve_provider() == "minimax"

    def test_auto_detect_minimax_key(self):
        env = {"MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_provider() == "minimax"

    def test_auto_detect_openai_key(self):
        env = {"OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_provider() == "openai"

    def test_minimax_takes_priority_over_openai(self):
        env = {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "oai-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_provider() == "minimax"

    def test_fallback_to_openai_when_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_provider() == "openai"

    def test_explicit_provider_overrides_keys(self):
        env = {"EVAL_LLM_PROVIDER": "openai", "MINIMAX_API_KEY": "mm-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_provider() == "openai"


# ---------------------------------------------------------------------------
# Temperature clamping
# ---------------------------------------------------------------------------


class TestClampTemperature:
    """Tests for temperature clamping per provider."""

    def test_minimax_clamps_zero(self):
        assert _clamp_temperature(0.0, "minimax") == 0.01

    def test_minimax_keeps_valid(self):
        assert _clamp_temperature(0.6, "minimax") == 0.6

    def test_minimax_clamps_above_one(self):
        assert _clamp_temperature(1.5, "minimax") == 1.0

    def test_openai_passes_through(self):
        assert _clamp_temperature(0.0, "openai") == 0.0
        assert _clamp_temperature(2.0, "openai") == 2.0

    def test_none_temperature(self):
        assert _clamp_temperature(None, "minimax") is None


# ---------------------------------------------------------------------------
# GPTEvaluation construction
# ---------------------------------------------------------------------------


class TestGPTEvaluationInit:
    """Tests for GPTEvaluation initialization."""

    @patch("gpt_eval.OpenAI")
    def test_creates_openai_provider(self, mock_openai_cls):
        evaluator = GPTEvaluation(provider="openai", api_key="test-key")
        assert evaluator.provider == "openai"
        assert evaluator.default_model == "gpt-3.5-turbo"
        mock_openai_cls.assert_called_once_with(api_key="test-key")

    @patch("gpt_eval.OpenAI")
    def test_creates_minimax_provider(self, mock_openai_cls):
        evaluator = GPTEvaluation(provider="minimax", api_key="test-key")
        assert evaluator.provider == "minimax"
        assert evaluator.default_model == "MiniMax-M2.7"
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )

    @patch("gpt_eval.OpenAI")
    def test_custom_base_url(self, mock_openai_cls):
        evaluator = GPTEvaluation(
            provider="minimax",
            api_key="test-key",
            base_url="https://custom.endpoint/v1",
        )
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom.endpoint/v1",
        )

    @patch("gpt_eval.OpenAI")
    def test_custom_model(self, mock_openai_cls):
        evaluator = GPTEvaluation(
            provider="minimax", api_key="test-key", model="MiniMax-M2.7-highspeed"
        )
        assert evaluator.default_model == "MiniMax-M2.7-highspeed"

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            GPTEvaluation(provider="unsupported", api_key="key")

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                GPTEvaluation(provider="minimax")

    @patch("gpt_eval.OpenAI")
    def test_reads_api_key_from_env(self, mock_openai_cls):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=False):
            evaluator = GPTEvaluation(provider="minimax")
            mock_openai_cls.assert_called_once_with(
                api_key="env-key",
                base_url="https://api.minimax.io/v1",
            )


# ---------------------------------------------------------------------------
# GPTEvaluation call_chatgpt
# ---------------------------------------------------------------------------


class TestCallChatGPT:
    """Tests for the call_chatgpt method."""

    @patch("gpt_eval.OpenAI")
    def test_uses_default_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="75"))]
        mock_response.usage = MagicMock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response

        evaluator = GPTEvaluation(provider="minimax", api_key="test-key")
        reply, tokens = evaluator.call_chatgpt([{"role": "user", "content": "test"}])

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "MiniMax-M2.7"
        assert reply == "75"
        assert tokens == 100

    @patch("gpt_eval.OpenAI")
    def test_minimax_temperature_clamped(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="80"))]
        mock_response.usage = MagicMock(total_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response

        evaluator = GPTEvaluation(provider="minimax", api_key="test-key")
        evaluator.call_chatgpt([{"role": "user", "content": "test"}])

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.6  # 0.6 is already valid


# ---------------------------------------------------------------------------
# GPTEvaluation forward
# ---------------------------------------------------------------------------


class TestForward:
    """Tests for the forward method."""

    @patch("gpt_eval.OpenAI")
    def test_forward_returns_score(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="82"))]
        mock_response.usage = MagicMock(total_tokens=200)
        mock_client.chat.completions.create.return_value = mock_response

        evaluator = GPTEvaluation(provider="openai", api_key="test-key")
        result = evaluator.forward(("my answer", "correct answer"))
        assert result == "82"

    @patch("gpt_eval.OpenAI")
    def test_prepare_chatgpt_message(self, mock_openai_cls):
        evaluator = GPTEvaluation(provider="openai", api_key="test-key")
        messages = evaluator.prepare_chatgpt_message("test prompt")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test prompt"


# ---------------------------------------------------------------------------
# PROVIDER_CONFIGS integrity
# ---------------------------------------------------------------------------


class TestProviderConfigs:
    """Tests for PROVIDER_CONFIGS structure."""

    def test_openai_config(self):
        cfg = PROVIDER_CONFIGS["openai"]
        assert cfg["env_key"] == "OPENAI_API_KEY"
        assert cfg["default_model"] == "gpt-3.5-turbo"

    def test_minimax_config(self):
        cfg = PROVIDER_CONFIGS["minimax"]
        assert cfg["env_key"] == "MINIMAX_API_KEY"
        assert cfg["base_url"] == "https://api.minimax.io/v1"
        assert cfg["default_model"] == "MiniMax-M2.7"

    def test_minimax_base_url_not_minimax_chat(self):
        """Ensure we don't use the deprecated api.minimax.chat domain."""
        assert "minimax.chat" not in PROVIDER_CONFIGS["minimax"]["base_url"]


# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------


class TestStripThinkTags:
    """Tests for _strip_think_tags helper."""

    def test_strips_think_block(self):
        content = "<think>\nSome reasoning here.\n</think>\n\n78"
        assert _strip_think_tags(content) == "78"

    def test_no_think_tags(self):
        assert _strip_think_tags("82") == "82"

    def test_none_input(self):
        assert _strip_think_tags(None) is None

    def test_empty_string(self):
        assert _strip_think_tags("") == ""

    def test_multiline_think(self):
        content = "<think>\nLine 1\nLine 2\n</think>\n\nResult text"
        assert _strip_think_tags(content) == "Result text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
