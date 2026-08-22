"""Request-scoped context variables for distributed tracing and audit trails.

The ``request_id`` variable is set by the correlation-ID middleware in
``main.py`` and is readable from any sync or async code during the request
lifecycle — for example, to tag prediction audit logs with the originating
request.
"""

from __future__ import annotations

from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="")
