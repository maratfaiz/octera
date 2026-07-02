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

        attempts = [t for t in _attempts[key] if now - t < window_seconds]
        if len(attempts) >= max_attempts:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Слишком много попыток. Попробуйте снова через минуту.",
            )
        attempts.append(now)
        _attempts[key] = attempts

    return dependency
