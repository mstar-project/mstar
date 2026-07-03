import os
from concurrent.futures import Future


class EventWakeup:
    def __init__(self):
        self.event = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    def _wake(self, _fut): # runs on whatever thread finished the future
        os.eventfd_write(self.event, 1) # one syscall, thread-safe, async-signal-safe

    def register_future(self, future: Future):
        if future.done():
            return
        future.add_done_callback(self._wake)

    def register_futures(self, futures):
        [self.register_future(fut) for fut in futures]

    @property
    def fd(self):
        return self.event

    def drain(self):
        try:
            os.eventfd_read(self.event)
        except BlockingIOError:
            pass
