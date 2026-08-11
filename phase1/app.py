"""Phase 1 service skeleton for agent-system-a (architecture.md §17 Phase 1
deliverable: "5 service skeletons + health checks"). No chat/orchestration
logic yet -- that is Phase 5 (Planner). This exists only to prove the
container builds, binds, and is health-checkable inside Docker Compose.

The real Phase 0 protocol code (phase0/graph_client.py) is untouched and
preserved; this skeleton does not replace or mock it.
"""

from fastapi import FastAPI

app = FastAPI(title="agent-system-a (planner-a) — Phase 1 skeleton")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "agent-system-a"}
