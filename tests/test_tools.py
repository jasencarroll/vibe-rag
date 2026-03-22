"""Smoke tests for tool registration."""
from vibe_rag.server import mcp


def test_all_tools_registered():
    tool_names = set()
    # FastMCP stores tools - access may vary by version
    # Try common attribute patterns
    if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
        tool_names = set(mcp._tool_manager._tools.keys())
    elif hasattr(mcp, '_tools'):
        tool_names = set(mcp._tools.keys())

    expected = {"search_code", "search_memory", "remember", "forget", "index_project", "ingest_doc"}
    assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}. Found: {tool_names}"
