"""FakeLLM behavior tests.

The FakeLLM is the only LLM that runs in the test suite; pinning its
semantics keeps the agent tests reliable.
"""
from __future__ import annotations

import pytest

from src.llm import FakeLLM


class TestFakeLLM:
    def test_responses_queue_returned_in_order(self):
        llm = FakeLLM(responses=["a", "b"])
        assert llm.complete("sys", "u1").text == "a"
        assert llm.complete("sys", "u2").text == "b"

    def test_responses_queue_exhausted_raises(self):
        llm = FakeLLM(responses=["only one"])
        llm.complete("s", "u")
        with pytest.raises(RuntimeError, match="exhausted"):
            llm.complete("s", "u")

    def test_responder_callable_is_used(self):
        llm = FakeLLM(responder=lambda system, user: f"echo:{user}")
        out = llm.complete("sys", "hello")
        assert out.text == "echo:hello"

    def test_calls_are_recorded(self):
        llm = FakeLLM(responses=["r"])
        llm.complete("the system prompt", "the user prompt")
        assert llm.calls == [("the system prompt", "the user prompt")]

    def test_must_provide_exactly_one_of_responses_or_responder(self):
        with pytest.raises(ValueError):
            FakeLLM()  # type: ignore[call-arg]
        with pytest.raises(ValueError):
            FakeLLM(responses=["x"], responder=lambda s, u: "y")

    def test_token_counts_are_non_zero(self):
        llm = FakeLLM(responses=["reasonable length"])
        r = llm.complete("system", "long user prompt with words")
        assert r.tokens_in >= 1
        assert r.tokens_out >= 1
