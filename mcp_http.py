"""HTTP (Streamable HTTP) entry point for the leads MCP server, with Bearer auth.

Set the bearer token via the MCP_GATE environment variable, then run:
    python mcp_http.py

Endpoint (MCP Streamable HTTP transport):
    POST http://host:8000/mcp   with header   Authorization: Bearer <the-token>

The /health path is unauthenticated for liveness probes.
"""

import os
import leads_server  # noqa: F401 — defines leads_server.mcp with the 3 tools

from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

MCP_GATE = os.environ.get("MCP_GATE", "")


def _allowed_hosts():
    raw = os.environ.get("MCP_ALLOWED_HOSTS", "")
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    for h in ("localhost", "127.0.0.1"):
        if h not in hosts:
            hosts.append(h)
    return hosts


# DNS-rebinding protection: allow only the hosts we actually serve.
# Behind NPM the Host header is the public domain, so it must be listed here.
leads_server.mcp.settings.transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=_allowed_hosts(),
)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, bearer: str):
        super().__init__(app)
        self.bearer = bearer

    async def dispatch(self, request, call_next):
        if request.url.path in ("/health", "/healthz"):
            return JSONResponse({"status": "ok"}, status_code=200)
        auth = request.headers.get("Authorization", "")
        if not self.bearer or auth != "Bearer " + self.bearer:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


mcp_app = leads_server.mcp.streamable_http_app()
if MCP_GATE:
    mcp_app.add_middleware(BearerAuthMiddleware, bearer=MCP_GATE)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_HTTP_PORT", "8000"))
    uvicorn.run(mcp_app, host=host, port=port, proxy_headers=True, forwarded_allow_ips="*")
