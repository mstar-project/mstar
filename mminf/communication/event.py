from concurrent.futures import Future
import os


class EventWakeup:
    def __init__(self):
        self.event = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
    
    def signal(self): # thread-safe, async-signal-safe
        """Wake any thread blocked in ``wait_for_work`` on this event's fd.

        Producers on other threads call this after enqueuing work onto a
        plain ``queue.Queue`` (which a poll on the socket fd can't observe)
        so the polling consumer wakes immediately instead of waiting out its
        timeout."""
        os.eventfd_write(self.event, 1) # one syscall

    def _wake(self, _fut): # runs on whatever thread finished the future
        self.signal()

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