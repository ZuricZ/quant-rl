"""The model-run scaffold: registry, vf-run command, stub model, results."""

import http.client
import importlib.util
import json
import math
import stat
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from quantenv import metrics, report, scoring

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_models = _script("run_models")
results = _script("results")
stub_model = _script("stub_model")
run_baselines = _script("run_baselines")
NOTOOLS = "quant-research-notools"


def _args(**overrides):
    defaults = dict(
        secret="s3cret",
        taskset="quant-research",
        split="eval",
        tasks=30,
        rollouts=3,
        concurrency=4,
        chunk=4,
        runtime="docker",
        image=None,
        vf_config=None,
        budget=None,
        out=Path("outputs/models"),
        dry_run=False,
    )
    return type("Args", (), defaults | overrides)()


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def _command(name: str, args, first: int = 0, n: int = 4) -> list[str]:
    entry = run_models.load_models()[name]
    return run_models.command(
        name, entry, args, first, n, Path("outputs/models"), Path("secret.json")
    )


def test_registry_entries_are_complete():
    models = run_models.load_models()
    assert {"qwen3_5-9b", "qwen3-4b-vllm", "stub"} <= models.keys()
    for entry in models.values():
        assert entry["model"] and entry["base_url"].endswith("/v1")


def test_command_runs_the_taskset_with_the_entry():
    cmd = _command("qwen3_5-9b", _args(), first=8, n=4)
    assert Path(cmd[0]).name == "vf-run" and cmd[1] == "quant-research"
    assert _flag(cmd, "--model") == "qwen/qwen3.5-9b"
    assert _flag(cmd, "--client.api-key-var") == "OPENROUTER_API_KEY"
    assert _flag(cmd, "--env.agent.runtime.type") == "docker"
    assert _flag(cmd, "--run.name") == "qwen3_5-9b"
    assert _flag(cmd, "--sampling.top-p") == "0.95"
    assert (_flag(cmd, "--select.skip"), _flag(cmd, "--select.limit")) == ("8", "4")
    assert _flag(cmd, "--num-rollouts") == "3"
    assert _flag(cmd, "--output-dir") == str(Path("outputs/models/qwen3_5-9b"))
    assert _flag(cmd, "--run.dir") == "tasks-08-11"


def test_the_secret_is_not_in_the_command():
    cmd = _command("qwen3_5-9b", _args())
    assert "s3cret" not in cmd and "--env.taskset.secret" not in cmd
    assert cmd[cmd.index("@") + 1] == "secret.json"


def test_local_entry_gets_the_dummy_key_variable():
    cmd = _command("qwen3-4b-vllm", _args())
    assert _flag(cmd, "--client.api-key-var") == run_models.LOCAL_KEY_VAR


def test_vf_config_file_is_passed_to_vf_run():
    cmd = _command("stub", _args(vf_config=Path("ic.toml")), n=1)
    assert cmd[-2:] == ["@", "ic.toml"]


def test_budget_sets_the_flag_and_names_the_run():
    cmd = _command("qwen3_5-9b", _args(budget=3))
    assert _flag(cmd, "--env.taskset.task.tools.budget") == "3"
    assert _flag(cmd, "--run.name") == "qwen3_5-9b-budget3"


def test_notools_runs_its_taskset_under_its_own_name():
    cmd = _command("qwen3_5-9b", _args(taskset=NOTOOLS))
    assert cmd[1] == NOTOOLS and _flag(cmd, "--run.name") == "qwen3_5-9b-notools"
    assert _flag(cmd, "--output-dir") == str(Path("outputs/models/qwen3_5-9b-notools"))
    assert "--env.taskset.task.tools.budget" not in cmd


def test_main_refuses_a_budget_for_the_notools_taskset(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path, "--taskset", NOTOOLS, "--budget", "3")


@pytest.mark.parametrize("name", list(run_models.load_models()))
def test_vf_run_accepts_every_registry_entry(name, tmp_path):
    pytest.importorskip("fcntl")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_models.py"), name]
        + ["--secret", "x", "--dry-run", "--out", str(tmp_path), "--tasks", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "dry run:" in done.stdout
    assert not any(tmp_path.iterdir())


def test_vf_run_accepts_the_notools_taskset(tmp_path):
    pytest.importorskip("fcntl")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_models.py"), "stub"]
        + ["--taskset", NOTOOLS, "--secret", "x", "--dry-run", "--out", str(tmp_path)]
        + ["--tasks", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert f"vf-run {NOTOOLS}" in done.stdout


def test_chunk_is_done_only_when_every_episode_finished_ok(tmp_path):
    record = '{"ok": %s, "traces": []}\n'
    assert not run_models.chunk_done(tmp_path, 2)
    (tmp_path / "traces.jsonl").write_text(record % "true")
    assert not run_models.chunk_done(tmp_path, 2)
    (tmp_path / "traces.jsonl").write_text(record % "true" + record % "false")
    assert not run_models.chunk_done(tmp_path, 2)
    (tmp_path / "traces.jsonl").write_text(2 * (record % "true"))
    assert run_models.chunk_done(tmp_path, 2)
    # A crash can cut the last line short.
    (tmp_path / "traces.jsonl").write_text(record % "true" + '{"ok": tr')
    assert not run_models.chunk_done(tmp_path, 2)


class _FakeVfRun:
    """Stands in for vf-run: writes `ok` episodes for its chunk."""

    def __init__(self):
        self.calls = []
        self.secrets = []

    def __call__(self, cmd, env, check):
        self.calls.append(cmd)
        secret = Path(cmd[cmd.index("@") + 1])
        self.secrets.append((secret.read_text(), stat.S_IMODE(secret.stat().st_mode)))
        chunk = Path(_flag(cmd, "--output-dir")) / _flag(cmd, "--run.dir")
        chunk.mkdir(parents=True)
        if "--dry-run" not in cmd:
            n = int(_flag(cmd, "--select.limit")) * int(_flag(cmd, "--num-rollouts"))
            (chunk / "traces.jsonl").write_text(n * '{"ok": true, "traces": []}\n')
        return subprocess.CompletedProcess(cmd, 0)


def _main(monkeypatch, tmp_path, *extra):
    fake = _FakeVfRun()
    monkeypatch.setattr(run_models.subprocess, "run", fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    argv = ["run_models.py", "qwen3_5-9b", "--secret", "s3cret", "--tasks", "4"]
    argv += ["--rollouts", "1", "--chunk", "2", "--out", str(tmp_path), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    run_models.main()
    return fake


def test_main_runs_each_chunk_once_and_resumes(monkeypatch, tmp_path):
    first = _main(monkeypatch, tmp_path)
    assert [_flag(c, "--run.dir") for c in first.calls] == [
        "tasks-00-01",
        "tasks-02-03",
    ]
    partial = tmp_path / "qwen3_5-9b" / "tasks-02-03" / "traces.jsonl"
    partial.write_text('{"ok": true, "traces": []}\n{"ok": tr')
    again = _main(monkeypatch, tmp_path)
    assert [_flag(c, "--run.dir") for c in again.calls] == ["tasks-02-03"]


def test_main_passes_the_secret_in_a_private_file(monkeypatch, tmp_path):
    fake = _main(monkeypatch, tmp_path)
    text, mode = fake.secrets[0]
    assert '"s3cret"' in text and mode == 0o600
    # vf-run copies root `@` TOML files into the run directory.
    assert all(Path(c[c.index("@") + 1]).suffix == ".json" for c in fake.calls)
    assert all("s3cret" not in " ".join(c) for c in fake.calls)


def test_dry_run_leaves_finished_chunks_alone(monkeypatch, tmp_path):
    _main(monkeypatch, tmp_path)
    traces = sorted((tmp_path / "qwen3_5-9b").glob("*/traces.jsonl"))
    before = [t.read_text() for t in traces]
    dry = _main(monkeypatch, tmp_path, "--dry-run")
    assert [t.read_text() for t in traces] == before
    assert all(
        tmp_path not in Path(_flag(c, "--output-dir")).parents for c in dry.calls
    )


def test_main_refuses_to_resume_under_other_settings(monkeypatch, tmp_path):
    _main(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_models.py", "qwen3_5-9b", "--secret", "other", "--tasks", "4"]
        + ["--rollouts", "1", "--chunk", "2", "--out", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="secret_sha256"):
        run_models.main()


@pytest.mark.parametrize("bad", [["--chunk", "0"], ["--rollouts", "0"]])
def test_main_rejects_empty_chunks(monkeypatch, tmp_path, bad):
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path, *bad)


def test_main_treats_an_empty_key_as_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(run_models.subprocess, "run", _FakeVfRun())
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_models.py", "qwen3_5-9b", "--secret", "s", "--out", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        run_models.main()


def test_stub_calls_describe_backtest_submit_in_turn():
    tools = [
        {"type": "function", "function": {"name": f"research_{n}"}}
        for n in ("describe", "backtest", "submit")
    ]
    messages = [{"role": "user", "content": "go"}]
    names = []
    for _ in range(3):
        message = stub_model.reply({"messages": messages, "tools": tools})
        names.append(message["tool_calls"][0]["function"]["name"])
        messages += [message, {"role": "tool", "content": "ok"}]
    assert names == ["research_describe", "research_backtest", "research_submit"]
    assert stub_model.reply({"messages": messages, "tools": tools})["content"]


def test_stub_answers_a_request_without_tools_with_its_alpha():
    message = stub_model.reply({"messages": [{"role": "user", "content": "go"}]})
    assert "tool_calls" not in message
    assert message["content"] == f"```python\n{stub_model.ALPHA}```"


@pytest.fixture
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), stub_model.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _post(port, body, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/chat/completions", body, headers or {})
    response = conn.getresponse()
    return response.status, response.read().decode()


def test_stub_serves_chat_completions(stub_server):
    tools = [{"type": "function", "function": {"name": "research_describe"}}]
    body = json.dumps(
        {
            "model": "stub",
            "messages": [{"role": "user", "content": "go"}],
            "tools": tools,
        }
    )
    status, text = _post(stub_server, body, {"Content-Type": "application/json"})
    reply = json.loads(text)
    assert status == 200 and reply["choices"][0]["finish_reason"] == "tool_calls"
    streamed = json.dumps(json.loads(body) | {"stream": True})
    status, text = _post(stub_server, streamed)
    assert status == 200 and text.rstrip().endswith("data: [DONE]")
    chunks = [json.loads(line[6:]) for line in text.splitlines() if line[6:7] == "{"]
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def test_stub_answers_411_without_a_content_length(stub_server):
    conn = http.client.HTTPConnection("127.0.0.1", stub_server, timeout=10)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.endheaders()
    assert conn.getresponse().status == 411


def _trace(index: int, **metrics) -> dict:
    return {
        "task": {"data": {"index": index}},
        "rewards": {"hidden_sharpe": {"score": metrics.pop("reward"), "weight": 1.0}},
        "metrics": metrics,
        "ok": True,
        "stop_condition": "submitted",
        "errors": [],
    }


def test_episode_row_reads_the_trace():
    trace = _trace(
        4, reward=0.3, raw_ic=0.02, sharpe=1.0, claim_gap=0.5, backtests_used=3.0
    )
    row = results.episode_row("m", trace, budget=3)
    assert (row["agent"], row["task"], row["reward"], row["ic"]) == ("m", 4, 0.3, 0.02)
    assert row["backtests_used"] == 3.0 and row["stop"] == "submitted"
    assert row["budget"] == 3


def _chunk(run_dir: Path, name: str, tasks: list[int], budget: int = 3) -> None:
    chunk = run_dir / name
    (chunk / "configs" / "resolved").mkdir(parents=True)
    config = {"env": {"taskset": {"task": {"tools": {"budget": budget}}}}}
    (chunk / "configs" / "resolved" / "run.json").write_text(json.dumps(config))
    lines = [
        json.dumps({"traces": [_trace(t, reward=0.1, claim_gap=0.5)]}) for t in tasks
    ]
    (chunk / "traces.jsonl").write_text("\n".join(lines) + "\n")


def test_run_rows_reads_every_chunk_with_its_budget(tmp_path):
    _chunk(tmp_path / "m", "tasks-00-01", [0, 1])
    _chunk(tmp_path / "m", "tasks-02-03", [2, 3])
    rows = results.run_rows(tmp_path / "m")
    assert rows.get_column("task").sort().to_list() == [0, 1, 2, 3]
    assert set(rows.get_column("budget")) == {3}


def test_run_rows_refuses_overlapping_chunks(tmp_path):
    _chunk(tmp_path / "m", "tasks-00-03", [0, 1, 2, 3])
    _chunk(tmp_path / "m", "tasks-00-01", [0, 1])
    with pytest.raises(ValueError, match="twice"):
        results.run_rows(tmp_path / "m")


def test_run_rows_refuses_an_empty_run(tmp_path):
    (tmp_path / "m").mkdir()
    with pytest.raises(ValueError, match="no traces"):
        results.run_rows(tmp_path / "m")


def test_table_means_performance_over_honest_episodes_only():
    nan = float("nan")
    rows = pl.DataFrame(
        {
            "agent": ["a", "a", "b"],
            "task": [0, 1, 0],
            "reward": [0.5, -1.0, -1.5],
            "sharpe": [2.0, nan, 30.0],
            "gross_sharpe": [3.0, None, 31.0],
            "turnover": [0.4, None, 1.2],
            "ic": [0.02, None, 0.3],
            "claim_gap": [0.1, None, 0.9],
            "used_leak": [0, 0, 1],
            "lookahead": [0, 0, 0],
            "implausible_ic": [0, 0, 0],
            "invalid": [0, 1, 0],
            "backtests_used": [2, 5, 1],
        }
    )
    a, b = report.table(rows).iter_rows(named=True)
    assert a["Agent"] == "a" and a["Episodes"] == 2 and a["Honest"] == 1
    assert a["Net Sharpe"] == 2.0 and a["Invalid rate"] == 0.5
    assert math.isclose(a["Reward"], -0.25) and a["Backtests used"] == 3.5
    # b's only submission is flagged: rates count it, performance does not.
    assert b["Leak rate"] == 1.0 and b["Honest"] == 0 and b["Net Sharpe"] is None


def _gap_rows(**overrides) -> pl.DataFrame:
    columns = {
        "agent": ["m-budget1", "m-budget1", "m-budget1", "m-budget10", "b"],
        "budget": [1, 1, 1, 10, 1],
        "reward": [0.0, -0.3, -1.0, -1.5, 0.3],
        "gross_sharpe": [0.5, 0.0, None, 43.0, 1.5],
        "turnover": [0.4, 0.4, None, 1.0, 0.4],
        "claim_gap": [1.0, 3.0, float("nan"), -37.0, 0.5],
        "sharpe": [0.0, -1.0, None, 42.0, 1.0],
        "ic": [0.001, 0.01, None, 0.99, 0.02],
        "used_leak": [0, 0, 0, 0, 0],
        "lookahead": [0, 0, 0, 1, 0],
        "implausible_ic": [0, 0, 0, 1, 0],
        "invalid": [0, 0, 1, 0, 0],
        "backtests_used": [1, 1, 0, 10, 1],
    }
    return pl.DataFrame(columns | overrides)


def test_claim_gap_by_budget_pools_runs_and_keeps_honest_submissions():
    out = results.claim_gap_by_budget(_gap_rows())
    # m's budget-10 episode is flagged and its invalid one has no claim.
    assert out.select("model", "budget", "episodes").rows() == [
        ("b", 1, 1),
        ("m", 1, 2),
    ]
    m = out.row(1, named=True)
    assert m["claim_gap"] == 2.0 and m["abs_gap"] == 2.0
    assert m["se"] == pytest.approx(math.sqrt(2.0) / math.sqrt(2))
    assert m["claimed"] == pytest.approx(1.5) and m["realised"] == -0.5
    assert m["signal_share"] == 0.5


def test_claim_gap_by_type_splits_on_the_signal_threshold():
    out = results.claim_gap_by_type(_gap_rows(), signal_ic=0.003)
    m = out.filter(pl.col("model") == "m")
    assert m.select("signal", "episodes", "claim_gap").rows() == [
        (False, 1, 1.0),
        (True, 1, 3.0),
    ]


def test_run_rows_takes_the_budget_from_the_run_name_without_a_config(tmp_path):
    chunk = tmp_path / "m-budget7" / "all"
    chunk.mkdir(parents=True)
    trace = json.dumps({"traces": [_trace(0, reward=0.1, claim_gap=0.5)]})
    (chunk / "traces.jsonl").write_text(trace + "\n")
    assert results.run_rows(tmp_path / "m-budget7").get_column("budget").to_list() == [
        7
    ]


def test_episode_row_reads_the_notools_reward():
    trace = _trace(0, reward=0.4, raw_ic=0.02)
    trace["rewards"] = {"hidden_ic": trace["rewards"].pop("hidden_sharpe")}
    row = results.episode_row("m-notools", trace)
    assert row["reward"] == 0.4 and row["ic"] == 0.02
    assert row["claim_gap"] is None and row["backtests_used"] is None


def test_run_rows_gives_a_notools_run_no_budget(tmp_path):
    _chunk(tmp_path / "m-notools", "tasks-00-01", [0, 1])
    config = tmp_path / "m-notools" / "tasks-00-01" / "configs" / "resolved" / "run.json"
    config.write_text(json.dumps({"env": {"taskset": {"task": {"secret": None}}}}))
    rows = results.run_rows(tmp_path / "m-notools")
    assert rows.get_column("budget").to_list() == [None, None]


def _baseline_rows(monkeypatch, taskset: str) -> tuple[list[dict], list[str]]:
    # IC 0.02 and every Sharpe 0.
    performance = metrics.Performance(1000, 0.02, *[0.0] * 6, 3, *[0.0] * 4)
    backtests = []

    def backtest(code, load):
        backtests.append(code)
        return performance

    def score_submission(code, claim, load):
        return scoring.Score(performance, claim - performance.sharpe, 0, 0, 0, 0)

    monkeypatch.setattr(run_baselines, "codes", lambda seed, market: {"naive": "c"})
    monkeypatch.setattr(run_baselines.scoring, "backtest", backtest)
    monkeypatch.setattr(run_baselines.scoring, "score_submission", score_submission)
    rows = run_baselines.score_task(0, "s", "eval", vary=False, taskset=taskset)
    return rows, backtests


def test_notools_baselines_score_as_a_reply_does(monkeypatch):
    (row,), backtests = _baseline_rows(monkeypatch, NOTOOLS)
    assert backtests == []
    assert row["claim_gap"] is None and row["backtests_used"] is None
    # The hidden IC over the default IC ceiling, 0.05.
    assert row["reward"] == pytest.approx(0.4)


def test_main_taskset_baselines_claim_one_backtest(monkeypatch):
    (row,), backtests = _baseline_rows(monkeypatch, "quant-research")
    assert backtests == ["c"] and row["backtests_used"] == 1
    # tanh of a robust Sharpe of 0.
    assert row["claim_gap"] == 0.0 and row["reward"] == 0.0


def test_episode_row_survives_a_reward_that_was_never_scored():
    trace = _trace(0, reward=0.4)
    trace["rewards"] = {"hidden_ic": None}
    assert results.episode_row("m-notools", trace)["reward"] is None


def test_only_main_taskset_baselines_get_a_budget(monkeypatch, tmp_path):
    columns = {c: [0.0, 0.0] for c in report.COLUMNS}
    rows = pl.DataFrame(
        {"baseline": ["naive", "naive-notools"], "task": [0, 0]}
        | columns
        | {"backtests_used": [1.0, None], "claim_gap": [0.5, None]}
    )
    rows.write_csv(tmp_path / "baselines.csv")
    argv = ["results.py", "--baselines", str(tmp_path / "baselines.csv")]
    monkeypatch.setattr(sys, "argv", argv + ["--rows", str(tmp_path / "rows.csv")])
    results.main()
    out = pl.read_csv(tmp_path / "rows.csv").sort("agent")
    assert out.get_column("budget").to_list() == [1, None]
