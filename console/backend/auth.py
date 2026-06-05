"""
Auth middleware — no-op locally, real provider when hosted.

Set AUTORESEARCH_AUTH_PROVIDER to activate:
  none    (default) — all requests pass through
  bearer  — validate Authorization: Bearer <token> against AUTORESEARCH_AUTH_TOKEN
  oidc    — validate OIDC JWT (requires additional deps; not yet implemented)
"""

from __future__ import annotations

import os
from fastapi import HTTPException, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


_PROVIDER = os.environ.get("AUTORESEARCH_AUTH_PROVIDER", "none")
_TOKEN = os.environ.get("AUTORESEARCH_AUTH_TOKEN", "")

# Paths that are always public
_PUBLIC = {"/api/health"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _PROVIDER == "none":
            return await call_next(request)

        if request.url.path in _PUBLIC:
            return await call_next(request)

        if _PROVIDER == "bearer":
            header = request.headers.get("authorization", "")
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or token != _TOKEN:
                return Response("Unauthorized", status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
            return await call_next(request)

        # OIDC and other providers: raise until implemented
        return Response("Auth provider not implemented", status_code=501)
