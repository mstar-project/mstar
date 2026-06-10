import logging

# Third-party libraries that log at INFO on essentially every operation and
# drown out mminf's own INFO logs. huggingface_hub fetches weights/configs
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
    per-request HTTP noise from weight downloads while mminf keeps a global INFO
    level. Call once after ``logging.basicConfig`` in each process.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
