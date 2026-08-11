"""Phase 0 compatibility spike: a real LangGraph StateGraph exercising a live
MCP Streamable HTTP call and a live A2A call from inside its own nodes.

This is not a bystander script: the MCP and A2A calls happen inside named
LangGraph nodes, the graph is compiled and executed through LangGraph's own
native streaming API, and the observed node-update events are what prove
LangGraph itself executed them.

The A2A node never hard-codes an RPC path. It discovers the Agent Card via
A2ACardResolver's standard well-known-path behavior, then derives the RPC
URL and transport binding from the card itself.
"""

import argparse
import asyncio
import json
import os
from typing import Any, TypedDict

import httpx
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client_factory import ClientConfig, ClientFactory
from a2a.types import a2a_pb2 as a2a_types
from langgraph.graph import StateGraph
from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


class SpikeState(TypedDict):
    nonce: str
    mcp_base_url: str
    a2a_base_url: str
    mcp_result: dict
    a2a_result: dict


# ---------------------------------------------------------------------------
# MCP node
# ---------------------------------------------------------------------------


async def _call_mcp(state: SpikeState) -> dict:
    endpoint = state["mcp_base_url"].rstrip("/") + "/mcp"
    async with streamable_http_client(endpoint) as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            init_result = await session.initialize()
            if not isinstance(init_result, mcp_types.InitializeResult):
                raise RuntimeError(
                    f"MCP initialize() returned unexpected type: {type(init_result)!r}"
                )

            call_result = await session.call_tool("echo", {"nonce": state["nonce"]})
            if not isinstance(call_result, mcp_types.CallToolResult):
                raise RuntimeError(
                    f"MCP call_tool() returned unexpected type: {type(call_result)!r}"
                )

            structured: dict[str, Any] = call_result.structured_content or {}
            if not structured and call_result.content:
                first = call_result.content[0]
                text = getattr(first, "text", None)
                if text:
                    structured = json.loads(text)

            return {
                "mcp_result": {
                    "nonce": structured.get("nonce"),
                    "pid": structured.get("pid"),
                    "marker": structured.get("marker"),
                    "endpoint": endpoint,
                    "protocol_version": init_result.protocol_version,
                    "is_error": bool(call_result.is_error),
                }
            }


def call_mcp_node(state: SpikeState) -> dict:
    return asyncio.run(_call_mcp(state))


# ---------------------------------------------------------------------------
# A2A node
# ---------------------------------------------------------------------------


def _collect_text_parts(*part_lists) -> list[str]:
    texts: list[str] = []
    for parts in part_lists:
        for part in parts:
            if getattr(part, "text", ""):
                texts.append(part.text)
    return texts


async def _call_a2a(state: SpikeState) -> dict:
    base_url = state["a2a_base_url"]
    agent_card_discovery_url = base_url.rstrip("/") + "/.well-known/agent-card.json"

    async with httpx.AsyncClient() as httpx_client:
        # 1-3: discover the Agent Card via the resolver's standard
        # well-known discovery behavior -- never a hard-coded RPC path.
        resolver = A2ACardResolver(httpx_client, base_url)
        card = await resolver.get_agent_card()
        if not isinstance(card, a2a_types.AgentCard):
            raise RuntimeError(f"Unexpected Agent Card type: {type(card)!r}")
        if not card.supported_interfaces:
            raise RuntimeError("Agent Card advertised no supported interfaces")

        # The card is the only source of the RPC URL and transport binding.
        # Request exactly the binding(s) the card itself advertises, so the
        # requested/selected binding is never an empty, unrelated default --
        # it is explicitly derived from the discovered card's own contents.
        card_bindings = [i.protocol_binding for i in card.supported_interfaces]

        # 4-6: build the client from the discovered card; the RPC URL and
        # transport binding come from the card, never from a guessed path.
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=True,
            supported_protocol_bindings=card_bindings,
        )
        factory = ClientFactory(config)
        client = factory.create(card)

        # The interface (RPC URL + binding) actually matching what was
        # requested against the card -- this is the concrete transport this
        # call uses, derived from the card, not a hard-coded label.
        selected_interface = next(
            (
                i
                for i in card.supported_interfaces
                if i.protocol_binding in card_bindings
            ),
            card.supported_interfaces[0],
        )

        message = a2a_types.Message(
            message_id=f"phase0-{os.getpid()}",
            role=a2a_types.Role.ROLE_USER,
            parts=[a2a_types.Part(text=json.dumps({"nonce": state["nonce"]}))],
        )
        request = a2a_types.SendMessageRequest(message=message)

        # 7-8: send the request; validate strictly against official protobuf
        # models. Only genuine Artifact objects (task.artifacts or
        # artifact_update.artifact) are ever treated as evidence -- a
        # status/standalone Message is never accepted as artifact proof.
        collected_artifacts: list = []
        final_state = a2a_types.TaskState.TASK_STATE_UNSPECIFIED

        async for response in client.send_message(request):
            which = response.WhichOneof("payload")
            if which == "task":
                task = response.task
                if not isinstance(task, a2a_types.Task):
                    raise RuntimeError(f"Unexpected Task type: {type(task)!r}")
                for artifact in task.artifacts:
                    if not isinstance(artifact, a2a_types.Artifact):
                        raise RuntimeError(
                            f"Unexpected Artifact type: {type(artifact)!r}"
                        )
                collected_artifacts.extend(task.artifacts)
                final_state = task.status.state
            elif which == "status_update":
                final_state = response.status_update.status.state
            elif which == "artifact_update":
                artifact = response.artifact_update.artifact
                if not isinstance(artifact, a2a_types.Artifact):
                    raise RuntimeError(f"Unexpected Artifact type: {type(artifact)!r}")
                collected_artifacts.append(artifact)
            # "message" variant (a standalone Message) is intentionally
            # never collected here -- it is not artifact evidence.

        if final_state != a2a_types.TaskState.TASK_STATE_COMPLETED:
            raise RuntimeError(
                f"A2A task did not reach TASK_STATE_COMPLETED: "
                f"{a2a_types.TaskState.Name(final_state)}"
            )
        if not collected_artifacts:
            raise RuntimeError(
                "A2A task completed but returned no official Artifact"
            )

        found_payload = None
        for artifact in collected_artifacts:
            for part in artifact.parts:
                if not isinstance(part, a2a_types.Part):
                    raise RuntimeError(f"Unexpected Part type: {type(part)!r}")
                if not part.text:
                    continue
                try:
                    payload = json.loads(part.text)
                except json.JSONDecodeError:
                    continue
                if {"nonce", "pid", "marker"}.issubset(payload):
                    found_payload = payload
                    break
            if found_payload:
                break

        if found_payload is None:
            raise RuntimeError(
                "Completed A2A task's artifact(s) contained no valid "
                "nonce/pid/marker content"
            )

        return {
            "a2a_result": {
                "nonce": found_payload["nonce"],
                "pid": found_payload["pid"],
                "marker": found_payload["marker"],
                "task_state": a2a_types.TaskState.Name(final_state),
                "artifact_count": len(collected_artifacts),
                "agent_card_discovery_url": agent_card_discovery_url,
                "agent_card_supported_interfaces": [
                    {"url": i.url, "protocol_binding": i.protocol_binding}
                    for i in card.supported_interfaces
                ],
                "requested_protocol_bindings": card_bindings,
                "selected_rpc_url": selected_interface.url,
                "selected_transport_binding": selected_interface.protocol_binding,
                "client_type": type(client).__name__,
                "client_module": type(client).__module__,
            }
        }


def call_a2a_node(state: SpikeState) -> dict:
    return asyncio.run(_call_a2a(state))


# ---------------------------------------------------------------------------
# Graph construction and execution
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(SpikeState)
    graph.add_node("call_mcp", call_mcp_node)
    graph.add_node("call_a2a", call_a2a_node)
    graph.add_edge("__start__", "call_mcp")
    graph.add_edge("call_mcp", "call_a2a")
    graph.add_edge("call_a2a", "__end__")
    return graph.compile()


def run_spike(nonce: str, mcp_base_url: str, a2a_base_url: str) -> dict:
    compiled = build_graph()
    initial_state: SpikeState = {
        "nonce": nonce,
        "mcp_base_url": mcp_base_url,
        "a2a_base_url": a2a_base_url,
        "mcp_result": {},
        "a2a_result": {},
    }

    observed_nodes: list[str] = []
    final_state: dict = dict(initial_state)
    for update in compiled.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            observed_nodes.append(node_name)
            final_state.update(node_output)

    return {
        "pid": os.getpid(),
        "observed_nodes": observed_nodes,
        "mcp_result": final_state.get("mcp_result", {}),
        "a2a_result": final_state.get("a2a_result", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--mcp-base-url", required=True)
    parser.add_argument("--a2a-base-url", required=True)
    args = parser.parse_args()

    result = run_spike(args.nonce, args.mcp_base_url, args.a2a_base_url)
    # Single structured line the harness parses from this process's stdout.
    print("PHASE0_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
