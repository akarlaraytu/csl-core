"""
ChimeraGuard x LlamaIndex integration demo.

This demo shows how to:
1. Compile and load a CSL policy.
2. Wrap LlamaIndex tools with ChimeraGuard.
3. Enforce policy before each tool call.
4. Handle ALLOW/BLOCK outcomes deterministically.

Run from repository root:
    python examples/integrations/llamaindex_agent_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from llama_index.core.tools import FunctionTool
except ImportError:
    print("Missing dependency: llama-index-core")
    print('Install with: pip install -e ".[llamaindex]"')
    raise SystemExit(1)


from chimera_core.language.compiler import CSLCompiler
from chimera_core.language.parser import parse_csl_file
from chimera_core.plugins.llamaindex import guard_tools
from chimera_core.runtime import ChimeraError, ChimeraGuard


def load_guard(policy_path: Path) -> ChimeraGuard:
    """Compile a CSL policy file and return a ready-to-use guard."""
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    constitution = parse_csl_file(str(policy_path))
    compiled = CSLCompiler().compile(constitution)
    return ChimeraGuard(compiled)


def build_tools() -> List[Any]:
    """Create raw LlamaIndex tools used by the demo."""

    def send_email(recipient_domain: str, pii_present: str) -> str:
        return f"EMAIL_SENT:{recipient_domain}:{pii_present}"

    def transfer_funds(amount: int, approval_token: str = "NO") -> str:
        return f"TRANSFER_OK:{amount}:approval={approval_token}"

    def query_db(db_table: str) -> str:
        return f"QUERY_OK:{db_table}"

    return [
        FunctionTool.from_defaults(
            fn=send_email,
            name="SEND_EMAIL",
            description="Send an email to a target domain.",
        ),
        FunctionTool.from_defaults(
            fn=transfer_funds,
            name="TRANSFER_FUNDS",
            description="Transfer funds after policy checks.",
        ),
        FunctionTool.from_defaults(
            fn=query_db,
            name="QUERY_DB",
            description="Query a logical table.",
        ),
    ]


def context_mapper(tool_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map tool kwargs into the policy schema.

    This keeps policy variable names stable even if tool signatures differ.
    """
    return {
        "amount": tool_kwargs.get("amount", 0),
        "approval_token": tool_kwargs.get("approval_token", "NO"),
        "recipient_domain": tool_kwargs.get("recipient_domain"),
        "pii_present": tool_kwargs.get("pii_present"),
        "db_table": tool_kwargs.get("db_table"),
    }


def build_guarded_tool_map(guarded_tools_iter: Iterable[Any]) -> Dict[str, Any]:
    """Index guarded tools by name for explicit scenario selection."""
    return {tool.name: tool for tool in guarded_tools_iter}


def run_sync_case(
    name: str,
    tool: Any,
    kwargs: Dict[str, Any],
    expected: str,
) -> Tuple[str, str, str, str]:
    """
    Execute one synchronous case.

    Returns: (case_name, expected, actual, detail)
    """
    try:
        result = tool.call(**kwargs)
        return (name, expected, "ALLOW", str(result))
    except ChimeraError as exc:
        return (name, expected, "BLOCK", f"{exc.constraint_name}")
    except Exception as exc:
        return (name, expected, "ERROR", str(exc))


async def run_async_case(
    name: str,
    tool: Any,
    kwargs: Dict[str, Any],
    expected: str,
) -> Tuple[str, str, str, str]:
    """Execute one asynchronous case through acall."""
    try:
        result = await tool.acall(**kwargs)
        return (name, expected, "ALLOW", str(result))
    except ChimeraError as exc:
        return (name, expected, "BLOCK", f"{exc.constraint_name}")
    except Exception as exc:
        return (name, expected, "ERROR", str(exc))


def print_summary(rows: List[Tuple[str, str, str, str]]) -> None:
    """Simple deterministic text summary."""
    print("\nLlamaIndex Guard Demo Results")
    print("-" * 90)
    print(f"{'Case':34} {'Expected':10} {'Actual':10} Detail")
    print("-" * 90)
    passed = 0
    for case_name, expected, actual, detail in rows:
        if expected == actual:
            passed += 1
        print(f"{case_name:34} {expected:10} {actual:10} {detail}")
    print("-" * 90)
    print(f"Passed: {passed}/{len(rows)}")


async def main() -> None:
    policy_path = PROJECT_ROOT / "examples" / "agent_tool_guard.csl"
    guard = load_guard(policy_path)

    raw_tools = build_tools()

    user_tools = build_guarded_tool_map(
        guard_tools(
            raw_tools,
            guard,
            context_mapper=context_mapper,
            inject={"user_role": "USER"},
            tool_field="tool",
        )
    )
    admin_tools = build_guarded_tool_map(
        guard_tools(
            raw_tools,
            guard,
            context_mapper=context_mapper,
            inject={"user_role": "ADMIN"},
            tool_field="tool",
        )
    )

    results: List[Tuple[str, str, str, str]] = []

    results.append(
        run_sync_case(
            "USER external PII email",
            user_tools["SEND_EMAIL"],
            {"recipient_domain": "EXTERNAL", "pii_present": "YES"},
            "BLOCK",
        )
    )
    results.append(
        run_sync_case(
            "ADMIN internal PII email",
            admin_tools["SEND_EMAIL"],
            {"recipient_domain": "INTERNAL", "pii_present": "YES"},
            "ALLOW",
        )
    )
    results.append(
        run_sync_case(
            "USER transfer attempt",
            user_tools["TRANSFER_FUNDS"],
            {"amount": 1000, "approval_token": "YES"},
            "BLOCK",
        )
    )
    results.append(
        await run_async_case(
            "ADMIN async DB query",
            admin_tools["QUERY_DB"],
            {"db_table": "CUSTOMERS"},
            "ALLOW",
        )
    )

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
