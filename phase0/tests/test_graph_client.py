"""Focused unit tests for the Phase 0 LangGraph client.

No network, no live servers. Tests graph structure and pure helper logic in
isolation; the live MCP/A2A round trip is exercised only by the full
cross-process harness (phase0/verify.py in the superproject), not here.
Run with `python -m pytest` from the service root so `phase0` resolves as a
top-level package.
"""

from a2a.types import a2a_pb2 as a2a_types
from langgraph.graph.state import CompiledStateGraph

from phase0.graph_client import _collect_text_parts, build_graph


def test_graph_compiles_with_expected_nodes():
    compiled = build_graph()
    assert isinstance(compiled, CompiledStateGraph)
    node_names = set(compiled.get_graph().nodes.keys())
    assert {"call_mcp", "call_a2a"}.issubset(node_names)


def test_collect_text_parts_gathers_all_text_across_lists():
    part_a = a2a_types.Part(text="hello")
    part_b = a2a_types.Part()  # no text set
    part_c = a2a_types.Part(text="world")

    texts = _collect_text_parts([part_a, part_b], [part_c])
    assert texts == ["hello", "world"]


def test_collect_text_parts_empty_when_no_text():
    part_a = a2a_types.Part()
    texts = _collect_text_parts([part_a])
    assert texts == []
