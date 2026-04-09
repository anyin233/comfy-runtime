"""Stub replacement for ComfyUI's ``server`` module.

Provides a lightweight :class:`PromptServer` that satisfies
``from server import PromptServer`` without importing aiohttp or
any real networking code.  Every method is a silent no-op.
"""

from __future__ import annotations

from typing import Optional


class _StubRoutes:
    """Stub that silently accepts @routes.get / @routes.post decorators."""

    def get(self, path):
        """No-op GET route decorator."""

        def decorator(func):
            return func

        return decorator

    def post(self, path):
        """No-op POST route decorator."""

        def decorator(func):
            return func

        return decorator


class PromptServer:
    """Minimal stand-in for ``server.PromptServer``.

    ``instance`` is set to a real stub object (not ``None``) after the
    class body so that code doing ``PromptServer.instance.send_sync(...)``
    does not crash.
    """

    instance: Optional[PromptServer] = None
    client_id = None
    last_node_id = None
    last_prompt_id = None
    routes = _StubRoutes()

    def send_sync(self, event, data, client_id=None):
        """No-op — swallows sync messages."""
        pass

    def send_progress_text(self, text, node_id):
        """No-op — swallows progress text."""
        pass

    def queue_updated(self):
        """No-op — swallows queue-update notifications."""
        pass


PromptServer.instance = PromptServer()
