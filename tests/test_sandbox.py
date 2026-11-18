"""Agent code in the sandbox: what it can reach, and exploits it must not pull off."""

import concurrent.futures
import dataclasses
import functools
import math
import os
import shutil
import time
from pathlib import Path

import pytest

from quantenv import baselines, metrics, reward, sandbox, scoring, walkforward
from quantenv.alpha import AlphaError, compile_alpha
from quantenv.sandbox import Sandbox
from quantenv.sources import synthetic

SMALL = synthetic.Config(
    n_assets=50, n_visible=200, n_hidden=400, no_leak_prob=0.2
)
SMALL_WF = walkforward.WalkForward(refit_every=50, window=50)
SEED = next(s for s in range(100) if synthetic.ground_truth(s, SMALL).leaky)
LEAK = synthetic.ground_truth(SEED, SMALL).leaky
VISIBLE = functools.partial(synthetic.visible_data, SEED, SMALL)
HIDDEN = functools.partial(synthetic.hidden_data, SEED, SMALL)

needs_root = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() != 0 or shutil.which("su") is None,
    reason="running as another user needs root on POSIX",
)


def _backtest(body: str, user: str | None = None) -> metrics.Performance:
    code = f"class S(Alpha):\n    def predict(self, df):\n{body}"
    # Generous: these tests are about isolation, and some alphas sleep.
    limit = scoring.TimeLimit(startup=60.0)
    return scoring.backtest(code, VISIBLE, SMALL_WF, limit, user=user)


def test_sandbox_matches_in_process_run():
    data = synthetic.generate(SEED, SMALL)
    code = baselines.weighted_code(data.truth.weights)
    local = walkforward.run(
        walkforward.Local(compile_alpha(code)), data.panel, SMALL_WF
    )
    boxed = walkforward.run(Sandbox(code, 30.0), data.panel, SMALL_WF)
    assert boxed.predictions.equals(local.predictions)
    assert boxed.engineered.equals(local.engineered)


def test_compile_error_is_an_alpha_error():
    with pytest.raises(AlphaError, match="code failed to run"):
        scoring.backtest("not python", VISIBLE, SMALL_WF)


def test_caching_across_runs_cannot_hide_the_leak():
    # Every check run is a fresh process, so a class-level cache filled in the
    # base run is empty in the permuted run.
    code = (
        "class Cache(Alpha):\n"
        "    memo = {}\n"
        "    def predict(self, df):\n"
        "        key = (df['timestamp'][0], df.height)\n"
        f"        return Cache.memo.setdefault(key, df[{LEAK!r}])\n"
    )
    score = scoring.score_submission(code, 0.3, HIDDEN, SMALL_WF)
    assert (
        score.used_leak == 1
        and reward.reward(dataclasses.asdict(score)) == reward.FLAGGED_REWARD
    )


def test_session_never_holds_unrealised_targets():
    # Walk every live object: no frame in the process may hold a target for
    # a row at or after the block being predicted.
    body = (
        "        import gc\n"
        "        first = df['timestamp'].min()\n"
        "        for o in gc.get_objects():\n"
        "            if isinstance(o, pl.DataFrame) and 'fwd_return' in o.columns:\n"
        "                assert o['timestamp'].max() < first, 'target in reach'\n"
        "        return pl.col('f00')\n"
    )
    _backtest(body)


def test_parent_environment_is_not_inherited(monkeypatch):
    monkeypatch.setenv("QUANTENV_TEST_SECRET", "x")
    body = (
        "        import os\n"
        "        assert 'QUANTENV_TEST_SECRET' not in os.environ\n"
        "        return pl.col('f00')\n"
    )
    _backtest(body)


@needs_root
def test_sandbox_user_cannot_read_private_files(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("seed")
    secret.chmod(0o600)
    body = f"        open({str(secret)!r}).read()\n        return pl.col('f00')\n"
    _backtest(body)  # As root it can.
    with pytest.raises(AlphaError, match="PermissionError"):
        _backtest(body, user="nobody")


@needs_root
def test_sandbox_user_leaves_nothing_behind():
    import pwd  # POSIX only.

    marker = "/tmp/quantenv-sandbox-test"
    body = (
        "        import subprocess\n"
        f"        open({marker!r}, 'w').write('x')\n"
        "        subprocess.Popen(['sleep', '600'], start_new_session=True)\n"
        "        return pl.col('f00')\n"
    )
    _backtest(body, user="nobody")
    assert not os.path.exists(marker)
    nobody = pwd.getpwnam("nobody").pw_uid
    deadline = time.monotonic() + 2.0
    while _live_processes(nobody) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _live_processes(nobody)


def _live_processes(uid: int) -> list[str]:
    """PIDs owned by `uid` that are not zombies."""
    live = []
    for proc in Path("/proc").iterdir():
        try:
            if proc.name.isdigit() and proc.stat().st_uid == uid:
                state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if state != "Z":
                    live.append(proc.name)
        except FileNotFoundError:
            continue
    return live


def test_dataset_seeds_depend_on_secret_split_and_index():
    a = synthetic.dataset_seed(0, "a", "eval")
    assert a == synthetic.dataset_seed(0, "a", "eval")
    assert a != synthetic.dataset_seed(0, "b", "eval")
    assert a != synthetic.dataset_seed(0, "a", "train")
    assert a != synthetic.dataset_seed(1, "a", "eval")
    assert a >= 2**64


_WALK = (
    "        import os\n"
    "        found = []\n"
    "        for root, dirs, files in os.walk('/'):\n"
    "            dirs[:] = [d for d in dirs if root != '/' or d not in ('proc', 'sys')]\n"
    "            {action}\n"
)


@needs_root
def test_no_writable_dir_carries_files_between_runs():
    # Run 1 drops a marker in every directory it can write; run 2 looks for
    # them everywhere. Any survivor could carry predictions between the
    # checks' runs.
    marker = "quantenv-persist-marker"
    write = (
        "try:\n"
        f"                open(os.path.join(root, {marker!r}), 'w').write('x')\n"
        "                found.append(root)\n"
        "            except OSError:\n"
        "                pass\n"
        "        open(os.path.join(os.getcwd(), "
        + repr(marker)
        + "), 'w').write('x')\n"
        "        assert found, 'nothing writable'\n"
        "        return pl.col('f00')\n"
    )
    look = (
        f"found += [root for f in files if f == {marker!r}]\n"
        "        assert not found, found\n"
        "        return pl.col('f00')\n"
    )
    _backtest(_WALK.format(action=write), user="nobody")
    _backtest(_WALK.format(action=look), user="nobody")


@needs_root
def test_sandbox_user_has_no_root_group():
    directory = Path("/tmp/quantenv-group-test")
    directory.mkdir(mode=0o755, exist_ok=True)
    directory.chmod(0o755)
    private = directory / "group-only.txt"
    private.write_text("seed")
    os.chown(private, 0, 0)
    private.chmod(0o640)
    try:
        body = f"        open({str(private)!r}).read()\n        return pl.col('f00')\n"
        with pytest.raises(AlphaError, match="PermissionError"):
            _backtest(body, user="nobody")
    finally:
        shutil.rmtree(directory)


@needs_root
def test_resource_limits_apply():
    body = (
        "        import resource\n"
        "        for limit in ('RLIMIT_AS', 'RLIMIT_NPROC', 'RLIMIT_FSIZE'):\n"
        "            soft, hard = resource.getrlimit(getattr(resource, limit))\n"
        "            assert hard != resource.RLIM_INFINITY, limit\n"
        "        return pl.col('f00')\n"
    )
    _backtest(body, user="nobody")


def test_an_unusable_sandbox_user_is_not_the_agents_fault():
    # A setup problem must fail the scoring, not score the agent as invalid.
    with pytest.raises(sandbox.SandboxError):
        scoring.score_submission(
            baselines.column_code("f00"), 0.1, HIDDEN, SMALL_WF, user="no-such-user-xyz"
        )


@needs_root
def test_concurrent_sessions_of_one_user_do_not_kill_each_other():
    # Each session ends by killing every process of the sandbox user, so two
    # at once would kill each other.
    short = "        return pl.col('f00')\n"
    long = "        import time\n        time.sleep(3)\n        return pl.col('f00')\n"
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        slow = pool.submit(_backtest, long, "nobody")
        time.sleep(4)  # The slow run is inside its sessions now.
        fast = pool.submit(_backtest, short, "nobody")
        results = [slow.result(), fast.result()]
    assert all(math.isfinite(r.sharpe) for r in results)
