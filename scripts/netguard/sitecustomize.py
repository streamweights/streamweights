"""Network guard for the offline gate. Put this directory on PYTHONPATH and every Python process
(spill and its children) refuses to open a network connection: the attempt is logged to the file
named by SPILL_NETLOG and raises NetworkAttempt, a BaseException, so no library's retry loop or
`except Exception` can swallow it. AF_UNIX sockets are allowed. This is a second layer: the gate's
proof is the container's --network none (or the macOS sandbox), and the log must stay empty."""

import os
import socket

_LOG = os.environ.get("SPILL_NETLOG")


class NetworkAttempt(BaseException):
    pass


def _deny(what, target):
    if _LOG:
        with open(_LOG, "a") as f:
            f.write(f"{os.getpid()} {what} {target!r}\n")
    raise NetworkAttempt(f"network attempt: {what} {target!r}")


if _LOG:
    _connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex

    def connect(self, address):
        if self.family != getattr(socket, "AF_UNIX", -1):
            _deny("connect", address)
        return _connect(self, address)

    def connect_ex(self, address):
        if self.family != getattr(socket, "AF_UNIX", -1):
            _deny("connect_ex", address)
        return _connect_ex(self, address)

    def getaddrinfo(host, *a, **k):
        _deny("getaddrinfo", host)

    def gethostbyname(host):
        _deny("gethostbyname", host)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    socket.getaddrinfo, socket.gethostbyname = getaddrinfo, gethostbyname
    socket.gethostbyname_ex = gethostbyname
