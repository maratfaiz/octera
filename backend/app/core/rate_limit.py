"""Minimal in-memory rate limiter.

Good enough for a single-process deployment; a real multi-worker/multi-node
production setup would need a shared store (e.g. Redis) instead.
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import HTTPException, Request, status

_attempts: dict[str, list[float]] = defaultdict(list)


def reset_rate_limits() -> None:
    """Clears all tracked attempts. Intended for use in tests."""
    _attempts.clear()


def rate_limit(key_prefix: str, max_attempts: int, window_seconds: int) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        key = f"{key_prefix}:{client_ip}"
        now = time.time()

        # Re-filter every key under this endpoint's own prefix against the
        # window, dropping ones that filter down to empty -- otherwise
        # _attempts only ever grows, one entry per distinct client IP ever
        # seen, for the life of the process (an inactive IP's entry is only
        # ever revisited, and thus only ever prunable, by that same IP
        # making a new request; storing `attempts` back only touched the
        # current key). Scoped to this prefix, not the whole dict, since
        # only same-prefix keys share this closure's window_seconds -- stays
        # correct if a second endpoint with a different window is added.
        # Only auth endpoints use this today, so a full per-prefix sweep on
        # every call is cheap.
        prefix = f"{key_prefix}:"
        for existing_key in [k for k in _attempts if k.startswith(prefix)]:
            filtered = [t for t in _attempts[existing_key] if now - t < window_seconds]
            if filtered:
                _attempts[existing_key] = filtered
            else:
                del _attempts[existing_key]

        attempts = _attempts[key]
        if len(attempts) >= max_attempts:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Слишком много попыток. Попробуйте снова через минуту.",
            )
        attempts.append(now)

    return dependency
