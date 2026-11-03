import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")
_runner_lock = threading.Lock()
_runner = None


class AsyncLoopRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="application-async-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot synchronously wait on the async loop thread.")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()


def get_async_runner() -> AsyncLoopRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = AsyncLoopRunner()
        return _runner
