"""Read-only MCP server: expose Langfuse traces to an AI agent via the REST API.

Works against any Langfuse (v2/v3) by calling the public REST API directly with
Basic auth — no langfuse SDK dependency. Read-only by design: only GET.

Env (set in the MCP client config):
    LANGFUSE_HOST          base URL of the self-hosted Langfuse
    LANGFUSE_PUBLIC_KEY    pk-lf-...
    LANGFUSE_SECRET_KEY    sk-lf-...
"""

import base64
import json
import os
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP

HOST = os.environ.get("LANGFUSE_HOST", "http://langfuse-langfuse-server-1:3000")
PK = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
SK = os.environ.get("LANGFUSE_SECRET_KEY", "")

mcp = FastMCP("langfuse-reader")


def _auth() -> str:
    return "Basic " + base64.b64encode((PK + ":" + SK).encode()).decode()


def _get(path: str) -> dict:
    req = urllib.request.Request(HOST + path, headers={"Authorization": _auth()})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": "HTTP %d: %s" % (e.code, e.read().decode()[:300])}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@mcp.tool()
def fetch_traces(limit: int = 20, page: int = 1) -> dict:
    """List recent Langfuse traces (one per MCP tool call).

    Returns a summary list: id, name (tool name), timestamp, input and output.
    Use fetch_trace(id) to drill into a single trace's observations.
    """
    data = _get("/api/public/traces?limit=%d&page=%d" % (limit, page))
    if isinstance(data, dict) and "error" in data:
        return data
    items = data.get("data", [])
    traces = [{
        "id": t.get("id"),
        "name": t.get("name"),
        "timestamp": t.get("timestamp"),
        "input": t.get("input"),
        "output": t.get("output"),
    } for t in items]
    return {"total": (data.get("meta") or {}).get("totalItems"), "traces": traces}


@mcp.tool()
def fetch_trace(trace_id: str) -> dict:
    """Fetch one trace by id, including its observations (input/output per call)."""
    data = _get("/api/public/traces/" + trace_id)
    if isinstance(data, dict) and "error" in data:
        return data
    obs = _get("/api/public/observations?traceId=" + trace_id + "&limit=100")
    observations = [{
        "id": o.get("id"),
        "name": o.get("name"),
        "type": o.get("type"),
        "input": o.get("input"),
        "output": o.get("output"),
    } for o in (obs.get("data", []) if isinstance(obs, dict) else [])]
    data["observations"] = observations
    return data


if __name__ == "__main__":
    mcp.run(transport="stdio")
