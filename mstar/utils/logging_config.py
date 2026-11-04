import atexit
import logging
import queue
from collections.abc import Callable
from logging.handlers import QueueHandler

from mstar.utils.threads import Thread

# Third-party libraries that log at INFO on essentially every operation and
# drown out mstar's own INFO logs. huggingface_hub fetches weights/configs
# through httpx (over httpcore/urllib3), which emits a line per HTTP request.
# These are kept at WARNING so the global level can stay INFO without the noise.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "filelock",
)


def quiet_noisy_loggers(level: int = logging.WARNING) -> None:
    """Raise chatty third-party loggers to ``level`` (default WARNING).

    Per-logger levels are independent of the root level, so this suppresses the
    per-request HTTP noise from weight downloads while mstar keeps a global INFO
    level. Call once after ``logging.basicConfig`` in each process.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def log_off_thread() -> Callable[[], None]:
    """Hand the root logger's records to a thread that writes them.

    A loop that logs on every request otherwise writes to stderr itself, under
    the handler's lock, and a stderr redirected to a file blocks while the
    kernel writes back dirty pages. Call once after ``logging.basicConfig``.
    Returns the drain, which also runs at exit and puts the handlers back.
    """
    root = logging.getLogger()
    handlers = root.handlers[:]
    records: queue.SimpleQueue[logging.LogRecord | None] = queue.SimpleQueue()
    # formats each record on the caller's thread, so its message and traceback
    # are fixed when it's logged
    to_writer = QueueHandler(records)
    for handler in handlers:
        root.removeHandler(handler)
    root.addHandler(to_writer)

    def write() -> None:
        while (record := records.get()) is not None:
            for handler in handlers:
                if record.levelno >= handler.level:
                    handler.handle(record)

    writer = Thread(target=write, daemon=True, name="log-writer")
    writer.start()

    def drain() -> None:
        records.put(None)
        writer.join()
        # whatever logs later in the exit writes for itself
        root.removeHandler(to_writer)
        for handler in handlers:
            root.addHandler(handler)

    atexit.register(drain)
    return drain
