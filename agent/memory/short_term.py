from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol


class ShortTermStore(Protocol):
    def push(self, sid: str, record: dict) -> None: ...
    def recent(self, sid: str, k: int) -> list[dict]: ...
    def clear(self, sid: str) -> None: ...


class InMemoryShortTermStore:
    def __init__(self, maxlen: int = 200) -> None:
        self._buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def push(self, sid: str, record: dict) -> None:
        self._buf[sid].append(record)

    def recent(self, sid: str, k: int) -> list[dict]:
        d = self._buf[sid]
        if k >= len(d):
            return list(d)
        return list(d)[-k:]

    def clear(self, sid: str) -> None:
        self._buf.pop(sid, None)


class RedisShortTermStore:
    """Optional backend. Requires `redis`; import is guarded so tests run without it."""

    def __init__(self, url: str, maxlen: int = 200, key_prefix: str = "st:") -> None:
        try:
            import redis  # type: ignore
        except ImportError as error:
            raise RuntimeError("redis backend requested but `redis` is not installed") from error
        self._r = redis.from_url(url)
        self.maxlen = maxlen
        self.prefix = key_prefix

    def _key(self, sid: str) -> str:
        return f"{self.prefix}{sid}"

    def push(self, sid: str, record: dict) -> None:
        import json

        self._r.rpush(self._key(sid), json.dumps(record, ensure_ascii=False))
        self._r.ltrim(self._key(sid), -self.maxlen, -1)

    def recent(self, sid: str, k: int) -> list[dict]:
        import json

        raw = self._r.lrange(self._key(sid), -k, -1)
        return [json.loads(item) for item in raw]

    def clear(self, sid: str) -> None:
        self._r.delete(self._key(sid))
