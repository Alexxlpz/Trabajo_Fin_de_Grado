import time
from typing import Optional

class HitTimer:
    def __init__(self, name: Optional[str] = None):
        self.name = name or "Timer"
        self._last = time.perf_counter()

    def hit(self) -> float:
        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now

        return elapsed

    def reset(self):
        self._last = time.perf_counter()