from __future__ import annotations

import pytest

from chimera_core.runtime import ChimeraError, ChimeraGuard
from chimera_core.plugins.llamaindex import ChimeraLlamaIndexGate, wrap_tool


def _has_llama_index() -> bool:
    try:
        import llama_index.core  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_llama_index(),
    reason="llama-index-core not installed; integration test skipped.",
)


def _extract_tool_value(result):
    if isinstance(result, str):
        return result
    if hasattr(result, "raw_output"):
        return str(result.raw_output)
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def test_llamaindex_gate_pass_through(compiled_agent_tool_guard):
    guard = ChimeraGuard(compiled_agent_tool_guard)
    gate = ChimeraLlamaIndexGate(
        guard,
        context_mapper=lambda x: x if isinstance(x, dict) else {"content": str(x)},
        inject={
            "tool": "NOOP",
            "user_role": "ADMIN",
            "recipient_domain": "INTERNAL",
            "target_table": "CUSTOMERS",
            "pii_present": "NO",
            "approval_token": "NO",
        },
    )

    x = {"hello": "world"}
    y = gate(x)
    assert y is x


def test_llamaindex_tool_wrapper_blocks_external_pii(compiled_agent_tool_guard):
    from llama_index.core.tools import FunctionTool

    guard = ChimeraGuard(compiled_agent_tool_guard)

    def send_email(recipient_domain: str, pii_present: str, user_role: str = "ADMIN") -> str:
        return "sent"

    tool = FunctionTool.from_defaults(
        fn=send_email,
        name="SEND_EMAIL",
        description="Test tool: send email",
    )

    def mapper(tool_kwargs: dict) -> dict:
        return {
            "tool": "SEND_EMAIL",
            "user_role": tool_kwargs.get("user_role", "ADMIN"),
            "pii_present": tool_kwargs.get("pii_present"),
            "recipient_domain": tool_kwargs.get("recipient_domain"),
        }

    guarded = wrap_tool(tool, guard, context_mapper=mapper)

    out = guarded.call(
        recipient_domain="INTERNAL",
        pii_present="YES",
        user_role="ADMIN",
    )
    assert _extract_tool_value(out) == "sent"

    with pytest.raises(ChimeraError) as excinfo:
        guarded.call(
            recipient_domain="EXTERNAL",
            pii_present="YES",
            user_role="ADMIN",
        )
    err = excinfo.value
    assert err.constraint_name == "no_external_email_with_pii"
    assert err.context.get("tool") == "SEND_EMAIL"
