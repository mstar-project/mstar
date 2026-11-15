"""``flashinfer.comm.comm_backend.TorchDistBackend``: holds its group; the
fusion workspace it would serve has no stand-in (see ``flashinfer.comm``)."""


class TorchDistBackend:
    def __init__(self, group=None):
        self.group = group
