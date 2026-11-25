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
- An empty environment, so the code sees none of the caller's variables,
  and a private working directory, also its TMPDIR and HOME.
- Optionally a dedicated unprivileged user, which needs root. The child
  takes that user's primary group and no supplementary groups, so it cannot
  read the caller's memory, environment or private files, including those
  readable by root's group. After each session every process of that user
  is killed and every file it owns, in any directory it can write, is
  removed, so nothing persists between runs.
- Resource limits on address space, processes and file size, set by the
  child itself before it loads the agent's code (POSIX only).
- One session at a time per sandbox user, across processes (an exclusive
  lock file), since each session's clean-up kills all of that user's
  processes.

Problems with the sandbox itself, such as an unknown user or missing
privileges, raise `SandboxError`, never `AlphaError`: they are not the
agent's fault.

Not covered: network access and System V IPC.

Run as `python -m quantenv.sandbox`; `Sandbox` starts it.
"""

import concurrent.futures
import contextlib
import dataclasses
import functools
import io
import os
import pickle
import shutil
import signal
import stat
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
_GiB = 2**30
# Not walked for writable directories: kernel interfaces, not storage.
_PSEUDO_FS = ("/proc", "/sys")


class SandboxError(RuntimeError):
    """The sandbox could not run, through no fault of the alpha code."""


@dataclasses.dataclass(frozen=True)
class Sandbox:
    """Runs one alpha's sessions in fresh child processes.

    Attributes:
        code: Alpha code.
        time_limit: Seconds per session, start-up included; the child is
            killed past it.
        user: Run the child as this user, or None for the caller's. The user
            should be dedicated to this: all its processes are killed and all
            its files removed after each session. Needs root.
        memory_bytes: RLIMIT_AS, the child's address space.
        max_processes: RLIMIT_NPROC: processes and threads of the user's,
            system-wide. polars starts about three threads per core in
            each process, and without `user` the caller's own processes
            (the scorer, its worker) count too.
        max_file_bytes: RLIMIT_FSIZE, the largest file the child can write.
    """

    code: str
    time_limit: float
    user: str | None = None
    memory_bytes: int = 16 * _GiB
    max_processes: int = dataclasses.field(
        default_factory=lambda: max(512, 32 * (os.cpu_count() or 1))
    )
    max_file_bytes: int = _GiB

    @property
    def limits(self) -> dict[str, int]:
        """The child's resource limits, by `resource` constant name."""
        return {
            "RLIMIT_AS": self.memory_bytes,
            "RLIMIT_NPROC": self.max_processes,
            "RLIMIT_FSIZE": self.max_file_bytes,
        }

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
            SandboxError: If the child cannot be started as configured.
        """
        with contextlib.ExitStack() as stack:
            if self.user is not None:
                stack.enter_context(_exclusive(self.user))
            yield stack.enter_context(self._child(timestamp))

    @contextlib.contextmanager
    def _child(self, timestamp: str) -> Iterator[Session]:
        workdir = tempfile.mkdtemp(prefix="quantenv-session-")
        identity = {}
        try:
            if self.user is not None:
                uid, gid = _ids(self.user)
                os.chown(workdir, uid, gid)
                identity = {"user": uid, "group": gid, "extra_groups": []}
            proc = subprocess.Popen(
                [sys.executable, "-I", "-m", "quantenv.sandbox"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={"TMPDIR": workdir, "HOME": workdir},
                cwd=workdir,
                start_new_session=True,
                **identity,
            )
        except OSError as e:
            shutil.rmtree(workdir, ignore_errors=True)
            raise SandboxError(f"cannot start the alpha process: {e!r}") from e
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        io_thread = concurrent.futures.ThreadPoolExecutor(1)
        child = _Child(proc, io_thread, time.monotonic() + self.time_limit)
        try:
            child.request(("start", self.code, timestamp, self.limits))
            yield child
        finally:
            io_thread.shutdown(wait=False, cancel_futures=True)
            _kill(proc)
            proc.wait()
            if self.user is not None:
                _clean_up(self.user)
            shutil.rmtree(workdir, ignore_errors=True)


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


def _ids(user: str) -> tuple[int, int]:
    """The uid and primary gid of `user`.

    Raises:
        SandboxError: If there is no such user.
    """
    import pwd  # POSIX only, like running as another user.

    try:
        entry = pwd.getpwnam(user)
    except KeyError as e:
        raise SandboxError(f"no sandbox user {user!r}") from e
    return entry.pw_uid, entry.pw_gid


@contextlib.contextmanager
def _exclusive(user: str) -> Iterator[None]:
    """Holds the lock that lets one session at a time run as `user`.

    The lock file is the caller's, mode 0600, so the sandbox user can
    neither open it nor have it swept as its own.
    """
    import fcntl  # POSIX only, like running as another user.

    path = os.path.join(tempfile.gettempdir(), f"quantenv-sandbox-{user}.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _permits(st: os.stat_result, uid: int, gid: int, mode: int) -> bool:
    """Whether `uid` with only group `gid` has every bit of `mode` (an rwx
    triple, e.g. 0o3 for write and search) on a file."""
    if st.st_uid == uid:
        shift = 6
    elif st.st_gid == gid:
        shift = 3
    else:
        shift = 0
    return (st.st_mode >> shift) & mode == mode


@functools.cache
def _writable_dirs(uid: int, gid: int) -> tuple[str, ...]:
    """Every directory `uid` can reach and create entries in.

    These are the only places a child running as `uid` can leave a file.
    Walks the whole file system once per process, pruning what the user
    cannot enter. Directories the user creates later lie inside these, so
    sweeping these removes them too.
    """
    found = []
    for root, dirs, _ in os.walk("/"):
        keep = []
        for name in dirs:
            path = os.path.join(root, name)
            if path in _PSEUDO_FS:
                continue
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if not stat.S_ISDIR(st.st_mode) or not _permits(st, uid, gid, 0o1):
                continue
            keep.append(name)
            if _permits(st, uid, gid, 0o3):
                found.append(path)
        dirs[:] = keep
    return tuple(found)


def _clean_up(user: str) -> None:
    """Kills every process of `user` and removes every file it owns."""
    subprocess.run(
        [sys.executable, "-I", "-c", "import os, signal; os.kill(-1, signal.SIGKILL)"],
        env={},
        user=user,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    uid, gid = _ids(user)
    for directory in _writable_dirs(uid, gid):
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.stat(follow_symlinks=False).st_uid != uid:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path, ignore_errors=True)
                else:
                    os.unlink(entry.path)
            except FileNotFoundError:
                continue


def _limit(limits: dict[str, int]) -> None:
    """Lowers this process's soft and hard resource limits (POSIX only)."""
    if os.name != "posix":
        return
    import resource

    for name, value in limits.items():
        resource.setrlimit(getattr(resource, name), (value, value))


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
                code, timestamp, limits = args
                _limit(limits)
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
