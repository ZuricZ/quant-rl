"""Runs agent code in a fresh, isolated process for every pipeline run.

Each session of `Sandbox` starts a new interpreter running this module. The
child compiles the alpha code and runs a `walkforward.Executor`, driven by
the trusted parent over its stdin and stdout:

- Requests go to the child pickled; the parent trusts nothing that comes
  back except data frames, read as Arrow IPC, never unpickled. A child can
  at most forge its own outputs, which the parent validates like any other.
- The child only receives what its session needs: the observable columns,
  then each block's newly realised targets just before that block.
- A fresh process per session: no state carries from one run to another, so
  the checks, which compare runs, compare independent runs.
- An empty environment, so the code sees none of the caller's variables.
- Optionally a dedicated unprivileged user, which needs root. That user
  cannot read the caller's memory, environment or private files. After each
  session every process of that user is killed and its files in the shared
  temporary directories are removed, so nothing persists between runs.

Not covered: network access and System V IPC.

Run as `python -m quantenv.sandbox`; `Sandbox` starts it.
"""

import concurrent.futures
import contextlib
import dataclasses
import io
import os
import pickle
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from typing import Any, BinaryIO

import polars as pl

from quantenv.alpha import AlphaError, AlphaTimeout, compile_alpha
from quantenv.walkforward import Block, Executor, Session

_OK, _ERROR = b"0", b"1"
_HEADER = struct.Struct("<cQ")  # Status, payload length.
_MAX_ERROR = 2000
_SHARED_TEMP = ("/tmp", "/var/tmp", "/dev/shm")


@dataclasses.dataclass(frozen=True)
class Sandbox:
    """Runs one alpha's sessions in fresh child processes.

    Attributes:
        code: Alpha code.
        time_limit: Seconds per session, start-up included; the child is
            killed past it.
        user: Run the child as this user, or None for the caller's. The user
            should be dedicated to this: all its processes are killed after
            each session. Needs root.
    """

    code: str
    time_limit: float
    user: str | None = None

    @contextlib.contextmanager
    def session(self, timestamp: str) -> Iterator[Session]:
        """A session in a new child process, killed and cleaned up on exit.

        Args:
            timestamp: Time column.

        Yields:
            The session.

        Raises:
            AlphaError: If the code fails to compile or a step fails.
            AlphaTimeout: If the session runs past its time limit.
        """
        proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "quantenv.sandbox"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd=tempfile.gettempdir(),
            user=self.user,
            start_new_session=True,
        )
        io_thread = concurrent.futures.ThreadPoolExecutor(1)
        child = _Child(proc, io_thread, time.monotonic() + self.time_limit)
        try:
            child.request(("start", self.code, timestamp))
            yield child
        finally:
            io_thread.shutdown(wait=False, cancel_futures=True)
            _kill(proc)
            proc.wait()
            if self.user is not None:
                _clean_up(self.user)


class _Child:
    """The parent's handle on a child: implements `Session` by messages."""

    def __init__(
        self,
        proc: subprocess.Popen,
        io_thread: concurrent.futures.ThreadPoolExecutor,
        deadline: float,
    ) -> None:
        self._proc = proc
        self._io = io_thread
        self._deadline = deadline

    def features(self, observable: pl.DataFrame) -> pl.DataFrame:
        return self.request(("features", observable))

    def use(self, engineered: pl.DataFrame) -> None:
        self.request(("use", engineered))

    def predict(self, block: Block, labels: pl.Series) -> pl.Series:
        return self.request(("predict", block, labels)).to_series()

    def request(self, message: tuple) -> pl.DataFrame:
        """Sends one message and waits for its reply, within the deadline."""
        exchange = self._io.submit(self._exchange, message)
        try:
            return exchange.result(timeout=max(self._deadline - time.monotonic(), 0))
        except concurrent.futures.TimeoutError as e:
            # Unblocks the exchange's read or write.
            self._proc.kill()
            raise AlphaTimeout(f"alpha ran past its time limit in {message[0]}") from e

    def _exchange(self, message: tuple) -> pl.DataFrame:
        try:
            pickle.dump(message, self._proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self._proc.stdin.flush()
            header = _read_exactly(self._proc.stdout, _HEADER.size)
            status, length = _HEADER.unpack(header)
            payload = _read_exactly(self._proc.stdout, length)
        except (OSError, EOFError) as e:
            raise AlphaError("alpha process died") from e
        if status != _OK:
            raise AlphaError(payload[:_MAX_ERROR].decode(errors="replace"))
        try:
            return pl.read_ipc(io.BytesIO(payload))
        except Exception as e:
            raise AlphaError(f"alpha process sent an invalid reply: {e!r}") from e


def _kill(proc: subprocess.Popen) -> None:
    """Kills the child and, on POSIX, every process left in its session."""
    if os.name != "posix":
        proc.kill()
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


def _read_exactly(stream: BinaryIO, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise EOFError("short read")
    return data


def _clean_up(user: str) -> None:
    """Kills every process of `user` and removes its shared temporary files."""
    subprocess.run(
        [sys.executable, "-I", "-c", "import os, signal; os.kill(-1, signal.SIGKILL)"],
        env={},
        user=user,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    import pwd  # POSIX only, like running as another user.

    uid = pwd.getpwnam(user).pw_uid
    for directory in {tempfile.gettempdir(), *_SHARED_TEMP}:
        with contextlib.suppress(FileNotFoundError), os.scandir(directory) as entries:
            for entry in entries:
                if entry.stat(follow_symlinks=False).st_uid != uid:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path, ignore_errors=True)
                else:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(entry.path)


def _reply(out: BinaryIO, status: bytes, payload: bytes) -> None:
    out.write(_HEADER.pack(status, len(payload)))
    out.write(payload)
    out.flush()


def _frame(value: Any) -> bytes:
    if value is None:
        value = pl.DataFrame()
    elif isinstance(value, pl.Series):
        value = value.to_frame()
    buffer = io.BytesIO()
    value.write_ipc(buffer)
    return buffer.getvalue()


def _serve(requests: BinaryIO, out: BinaryIO) -> None:
    """The child's loop: one reply per request until stdin closes."""
    executor = None
    while True:
        try:
            op, *args = pickle.load(requests)
        except EOFError:
            return
        try:
            if op == "start":
                code, timestamp = args
                executor = Executor(compile_alpha(code), timestamp)
                result = None
            else:
                result = getattr(executor, op)(*args)
            _reply(out, _OK, _frame(result))
        except AlphaError as e:
            _reply(out, _ERROR, str(e).encode())
        except Exception as e:
            _reply(out, _ERROR, f"{type(e).__name__}: {e}".encode())


def main() -> None:
    """Child entry point. Keeps the protocol pipe for itself and points the
    standard streams that agent code sees at /dev/null."""
    requests = os.fdopen(os.dup(sys.stdin.fileno()), "rb")
    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1):
        os.dup2(null, fd)
    sys.stdin = open(os.devnull)
    sys.stdout = open(os.devnull, "w")
    _serve(requests, out)


if __name__ == "__main__":
    main()
