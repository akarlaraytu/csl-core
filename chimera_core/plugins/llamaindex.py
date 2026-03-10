"""
CSL-Core - LlamaIndex Integration Plugin

Optional LlamaIndex guard plugin using the `ChimeraPlugin` architecture.
Provides policy-aware wrappers for agent/tool interactions without forcing
`llama_index` as a hard dependency at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ..runtime import ChimeraGuard
from .base import ChimeraPlugin, ContextMapper

if TYPE_CHECKING:
    try:
        from llama_index.core.tools.types import BaseTool as LlamaIndexBaseTool
    except ImportError:
        from llama_index.core.tools import BaseTool as LlamaIndexBaseTool
else:
    LlamaIndexBaseTool = Any


def _require_llamaindex() -> None:
    """Raises a clear, friendly error when LlamaIndex is unavailable."""
    try:
        import llama_index.core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex plugin requires `llama-index-core`. "
            "Install with: pip install llama-index-core"
        ) from exc


class ChimeraLlamaIndexGate(ChimeraPlugin):
    """
    Pass-through guard component for LlamaIndex pipelines.
    Normalizes input, verifies policy, then returns original input unchanged.
    """

    def __init__(
        self,
        guard_or_constitution: Any,
        *,
        context_mapper: Optional[ContextMapper] = None,
        inject: Optional[Dict[str, Any]] = None,
        enable_dashboard: bool = False,
    ):
        _require_llamaindex()
        if isinstance(guard_or_constitution, ChimeraGuard):
            constitution = guard_or_constitution.constitution
        else:
            constitution = guard_or_constitution

        super().__init__(
            constitution=constitution,
            enable_dashboard=enable_dashboard,
            context_mapper=context_mapper,
            title="LlamaIndex::Gate",
        )
        self.inject = inject or {}

    def process(self, input_data: Any) -> Any:
        return self.__call__(input_data)

    def __call__(self, input_data: Any) -> Any:
        self.run_guard(input_data, extra_context=self.inject)
        return input_data


def gate(
    guard: Any,
    *,
    context_mapper: Optional[ContextMapper] = None,
    inject: Optional[Dict[str, Any]] = None,
    enable_dashboard: bool = False,
) -> ChimeraLlamaIndexGate:
    """Helper to create a ChimeraLlamaIndexGate."""
    return ChimeraLlamaIndexGate(
        guard,
        context_mapper=context_mapper,
        inject=inject,
        enable_dashboard=enable_dashboard,
    )


class _ToolPlugin(ChimeraPlugin):
    """Internal helper to bridge LlamaIndex tool calls into ChimeraPlugin logic."""

    def process(self, input_data: Any) -> Any:
        return input_data


class GuardedLlamaIndexTool:
    """
    Wrapper that enforces policy before executing a LlamaIndex tool call.
    Uses composition to avoid framework inheritance/version coupling.
    """

    def __init__(
        self,
        tool: LlamaIndexBaseTool,
        plugin: _ToolPlugin,
        *,
        inject: Optional[Dict[str, Any]] = None,
        tool_field: Optional[str] = None,
    ):
        self.original_tool = tool
        self._plugin = plugin
        self._inject = inject or {}
        self._tool_field = tool_field

    @property
    def metadata(self) -> Any:
        return getattr(self.original_tool, "metadata", None)

    @property
    def name(self) -> str:
        metadata = self.metadata
        if metadata is not None and getattr(metadata, "name", None):
            return str(metadata.name)
        return str(getattr(self.original_tool, "name", "unknown"))

    @property
    def description(self) -> str:
        metadata = self.metadata
        if metadata is not None and getattr(metadata, "description", None):
            return str(metadata.description)
        return str(getattr(self.original_tool, "description", ""))

    def _build_extra_context(self) -> Dict[str, Any]:
        extra = self._inject.copy()
        if self._tool_field:
            extra[self._tool_field] = self.name
        return extra

    def _normalize_tool_input(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            return kwargs
        if args:
            return args[0]
        return {}

    def call(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = self._normalize_tool_input(*args, **kwargs)
        self._plugin.run_guard(tool_input, extra_context=self._build_extra_context())

        if hasattr(self.original_tool, "call"):
            return self.original_tool.call(*args, **kwargs)
        if callable(self.original_tool):
            return self.original_tool(*args, **kwargs)
        raise TypeError("Wrapped tool does not implement `call` and is not callable.")

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = self._normalize_tool_input(*args, **kwargs)
        self._plugin.run_guard(tool_input, extra_context=self._build_extra_context())

        if hasattr(self.original_tool, "acall"):
            return await self.original_tool.acall(*args, **kwargs)
        return self.call(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_tool, name)


def wrap_tool(
    tool: LlamaIndexBaseTool,
    guard: ChimeraGuard,
    *,
    context_mapper: Optional[ContextMapper] = None,
    inject: Optional[Dict[str, Any]] = None,
    tool_field: Optional[str] = None,
    enable_dashboard: bool = False,
) -> GuardedLlamaIndexTool:
    """Wrap a LlamaIndex tool with ChimeraGuard protection."""
    _require_llamaindex()
    plugin = _ToolPlugin(
        constitution=guard.constitution,
        enable_dashboard=enable_dashboard,
        context_mapper=context_mapper,
        title=f"Tool::{getattr(getattr(tool, 'metadata', None), 'name', getattr(tool, 'name', 'unknown'))}",
    )
    return GuardedLlamaIndexTool(
        tool,
        plugin,
        inject=inject,
        tool_field=tool_field,
    )


def guard_tools(
    tools: Iterable[LlamaIndexBaseTool],
    guard: ChimeraGuard,
    *,
    context_mapper: Optional[ContextMapper] = None,
    inject: Optional[Dict[str, Any]] = None,
    tool_field: Optional[str] = None,
    enable_dashboard: bool = False,
) -> List[GuardedLlamaIndexTool]:
    """Wrap multiple LlamaIndex tools at once."""
    return [
        wrap_tool(
            t,
            guard,
            context_mapper=context_mapper,
            inject=inject,
            tool_field=tool_field,
            enable_dashboard=enable_dashboard,
        )
        for t in tools
    ]
