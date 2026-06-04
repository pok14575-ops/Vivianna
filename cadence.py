from typing import Optional

_STRONG = frozenset('.!?…。！？')
_WEAK   = frozenset(',;:、')

MIN_CHARS      = 28   # flush at word boundary once buffer reaches this
WEAK_MIN_CHARS = 25   # flush on weak punctuation once buffer reaches this
HARD_CAP       = 90   # flush unconditionally to prevent unbounded growth


class CadenceBuffer:
    def __init__(self):
        self._buf: list[str] = []
        self._len: int = 0

    def push(self, token: str) -> Optional[str]:
        """
        Accumulate token. Returns a chunk to speak when a flush condition is met,
        otherwise returns None.

        Pipeline: stabilizer.iter_stream() → push() → output_bus.emit()
        """
        if not token:
            return None

        self._buf.append(token)
        self._len += len(token)
        last = token[-1]

        # Strong punctuation: flush immediately regardless of length
        if last in _STRONG:
            return self._drain()

        # Weak punctuation: flush only once there is enough content
        if last in _WEAK and self._len >= WEAK_MIN_CHARS:
            return self._drain()

        # Word boundary: flush once at or past MIN_CHARS
        if self._len >= MIN_CHARS and last in (' ', '\n'):
            return self._drain()

        # Hard cap: flush regardless to prevent runaway buffer
        if self._len >= HARD_CAP:
            return self._drain()

        return None

    def flush(self) -> Optional[str]:
        """Force-drain any remaining buffered content."""
        return self._drain() if self._buf else None

    def _drain(self) -> Optional[str]:
        text = ''.join(self._buf).strip()
        self._buf.clear()
        self._len = 0
        return text if text else None
