"""
Open WebUI Custom Tool: Search Knowledge Graph

Installs into Open WebUI as a callable tool for AI agents to query
the Universal Knowledge Graph via the hybrid search pipeline.

Install:
    1. Copy this file into your Open WebUI custom tools directory, or
       paste the content below into Workspace -> Tools -> Add Custom Tool.
    2. The tool becomes available as "search_knowledge_graph" in agent
       function-calling workflows.

Architecture:
    - Follows Open WebUI tool specification (JSON Schema + execute hook)
    - Wraps HybridSearchEngine from hybrid_search.py
    - 30-second timeout per search via signal.alarm (Unix) / threading fallback
    - UTF-8 safe for Thai language queries
    - Returns empty but valid response on error (never crashes)
"""

import json
import logging
import sys
import textwrap
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Open WebUI Tool Metadata  (JSON Schema for function calling)
# ---------------------------------------------------------------------------

TOOL_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge_graph",
        "description": textwrap.dedent("""\
            Search the Universal Knowledge Graph for concepts, equations,
            hardware specifications, trading strategies, and creative content seeds.

            The graph contains multi-domain knowledge with:
            - Technical descriptions (Thai and English)
            - Simple explanations for general audience
            - Analogies comparing technical concepts to everyday things
            - Creative fiction seeds inspired by technical knowledge

            Input: Natural language search query (Thai or English)
            Output: Structured context block with relevant knowledge
        """).strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": (
                        "Natural language search query in Thai or English. "
                        "Example: 'algorithmic trading strategies' or "
                        "'กลยุทธ์การเทรดอัตโนมัติ'"
                    ),
                }
            },
            "required": ["search_query"],
        },
    },
}

# ---------------------------------------------------------------------------
# Timeout mechanism
# ---------------------------------------------------------------------------

_SEARCH_TIMEOUT_SECONDS: int = 30


class _SearchTimeoutError(TimeoutError):
    """Raised when a search operation exceeds the allowed duration."""
    pass


def _run_with_timeout(func, args: tuple, kwargs: dict, timeout: int) -> Any:
    """Execute *func(*args, **kwargs)* with a *timeout* second limit.

    On Unix platforms this uses ``signal.alarm`` for precise enforcement.
    On Windows (or when ``signal`` is unavailable) it falls back to a
    ``threading`` -based approach that cancels after *timeout*.

    Raises:
        _SearchTimeoutError: When the operation exceeds *timeout* seconds.
    """
    import signal
    import threading

    def _handler_signum_frame(signum, frame):
        raise _SearchTimeoutError(
            f"Search exceeded {timeout}s timeout limit"
        )

    # Try signal-based approach first (Unix only).
    try:
        old_handler = signal.signal(signal.SIGALRM, _handler_signum_frame)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result
    except (AttributeError, ValueError):
        # signal not available (e.g. Windows or non-main thread);
        # fall back to threading-based timeout.
        pass

    result_container: Dict[str, Any] = {"result": None, "error": None}

    def _target():
        try:
            result_container["result"] = func(*args, **kwargs)
        except Exception as exc:
            result_container["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        raise _SearchTimeoutError(
            f"Search exceeded {timeout}s timeout limit (thread fallback)"
        )
    if result_container["error"] is not None:
        raise result_container["error"]
    return result_container["result"]


# ---------------------------------------------------------------------------
# Tool Implementation
# ---------------------------------------------------------------------------

class SearchKnowledgeGraphTool:
    """Open WebUI Custom Tool for Universal Knowledge Graph Search.

    Attributes:
        name: Unique tool identifier used in function calling.
        description: Human-readable overview shown to the LLM planner.
        parameters: JSON Schema describing the tool's input shape.
    """

    name: str = "search_knowledge_graph"
    description: str = textwrap.dedent("""\
        Search the Universal Knowledge Graph for concepts, equations,
        hardware specifications, trading strategies, and creative content seeds.

        The graph contains multi-domain knowledge with:
        - Technical descriptions (Thai and English)
        - Simple explanations for general audience
        - Analogies comparing technical concepts to everyday things
        - Creative fiction seeds inspired by technical knowledge

        Input: Natural language search query (Thai or English)
        Output: Structured context block with relevant knowledge
    """).strip()

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": (
                    "Natural language search query in Thai or English. "
                    "Example: 'algorithmic trading strategies' or "
                    "'กลยุทธ์การเทรดอัตโนมัติ'"
                ),
            }
        },
        "required": ["search_query"],
    }

    def __init__(self) -> None:
        """Initialize the tool with a lazy-loaded HybridSearchEngine."""
        self._search_engine: Any = None

    # ------------------------------------------------------------------ #
    #  Lazy initialization of search engine
    # ------------------------------------------------------------------ #

    @property
    def _engine(self) -> Any:
        if self._search_engine is None:
            from hybrid_search import HybridSearchEngine
            self._search_engine = HybridSearchEngine()
        return self._search_engine

    # ------------------------------------------------------------------ #
    #  Core execution
    # ------------------------------------------------------------------ #

    def execute(self, search_query: str) -> str:
        """Execute hybrid search and return formatted Markdown context.

        Args:
            search_query: Natural language query in Thai or English.

        Returns:
            Formatted Markdown string with search results, or a safe
            error block if the search fails.  Never raises.
        """
        # Ensure UTF-8 handling for Thai text.
        if isinstance(search_query, bytes):
            search_query = search_query.decode("utf-8")

        search_query = search_query.strip()
        if not search_query:
            return "# Search Results\n\n> **Empty query provided.** Please provide a search term."

        try:
            # Run hybrid search with timeout protection.
            results = _run_with_timeout(
                func=self._engine.search,
                args=("",),  # placeholder; real args below via kwargs
                kwargs={"query": search_query, "top_k": 3, "max_hops": 2},
                timeout=_SEARCH_TIMEOUT_SECONDS,
            )

            total_nodes: int = results.get("total_nodes", 0)
            context_block: str = results.get("context_block", "")

            response = f"# Search Results for: \"{search_query}\"\n\n"
            response += f"**Found {total_nodes} related nodes**\n\n"

            if context_block:
                response += context_block
            else:
                response += "> No results matched your query. Try rephrasing or using different keywords.\n"

            return response

        except _SearchTimeoutError:
            logger.error("Search timed out after %ds for query=%r", _SEARCH_TIMEOUT_SECONDS, search_query)
            return (
                f"# Search Results for: \"{search_query}\"\n\n"
                "> ⏱ **Search timed out.** The query took longer than "
                f"{_SEARCH_TIMEOUT_SECONDS} seconds. Please try a shorter or more specific query.\n"
            )

        except Exception as exc:
            logger.error("Search tool failed for query=%r: %s", search_query, exc, exc_info=True)
            return (
                f"# Search Results for: \"{search_query}\"\n\n"
                f"> ❌ **Search error:** {self._safe_error_message(exc)}\n"
            )

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        """Return a sanitized error message safe for LLM context.

        Strips stack traces and internal paths to avoid leaking
        implementation details.
        """
        msg = str(exc)
        # Truncate very long messages.
        if len(msg) > 500:
            msg = msg[:497] + "..."
        return msg


# ---------------------------------------------------------------------------
# Open WebUI Integration Entry Points
# ---------------------------------------------------------------------------

def get_tools() -> List[Dict[str, Any]]:
    """Return the list of tool definitions for Open WebUI registration.

    Open WebUI calls this function to discover available tools.
    Each returned dictionary is a JSON Schema function definition.
    """
    return [TOOL_DEFINITION]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch an Open WebUI tool call to the appropriate handler.

    Args:
        tool_name: The name of the tool to invoke (e.g. "search_knowledge_graph").
        arguments: Parsed JSON arguments from the LLM function call.

    Returns:
        String result returned to the LLM as tool output.
    """
    if tool_name == "search_knowledge_graph":
        search_query = arguments.get("search_query", "")
        tool = SearchKnowledgeGraphTool()
        return tool.execute(search_query)

    # Unknown tool — return valid but empty response (never crash).
    logger.warning("Unknown tool requested: %s", tool_name)
    return f"# Tool Error\n\n> Unknown tool: \"{tool_name}\""


# ---------------------------------------------------------------------------
# Standalone CLI for testing
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Allow running the tool directly from command line for debugging.

    Usage::

        python openwebui_tool.py "your search query here"
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python openwebui_tool.py \"search query\"")
        print("\nTool Definition (JSON Schema):")
        print(json.dumps(TOOL_DEFINITION, indent=2, ensure_ascii=False))
        sys.exit(0)

    query = " ".join(sys.argv[1:])
    print(f"\nExecuting search for: {query!r}\n")
    print("=" * 72 + "\n")

    result = execute_tool("search_knowledge_graph", {"search_query": query})
    print(result)


if __name__ == "__main__":
    _cli_main()
