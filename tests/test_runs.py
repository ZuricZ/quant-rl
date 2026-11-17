"""The model-run scaffold: registry, vf-run command, stub model, results."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from quantenv import report

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_models = _script("run_models")
results = _script("results")
stub_model = _script("stub_model")


def _args(**overrides):
    defaults = dict(
        secret="s3cret",
        split="eval",
        tasks=30,
        rollouts=3,
        concurrency=4,
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


def test_registry_entries_are_complete():
    models = run_models.load_models()
    assert {"qwen3_5-9b", "qwen3-4b-vllm", "stub"} <= models.keys()
    for entry in models.values():
        assert entry["model"] and entry["base_url"].endswith("/v1")


def test_command_runs_the_taskset_with_the_entry():
    entry = run_models.load_models()["qwen3_5-9b"]
    cmd = run_models.command("qwen3_5-9b", entry, _args(), first=8, n=4)
    assert cmd[:2] == ["vf-run", "quant-research"]
    assert _flag(cmd, "--model") == "qwen/qwen3.5-9b"
    assert _flag(cmd, "--client.api-key-var") == "OPENROUTER_API_KEY"
    assert _flag(cmd, "--env.taskset.secret") == "s3cret"
    assert _flag(cmd, "--env.agent.runtime.type") == "docker"
    assert _flag(cmd, "--run.name") == "qwen3_5-9b"
    assert _flag(cmd, "--sampling.top-p") == "0.95"
    assert (_flag(cmd, "--select.skip"), _flag(cmd, "--select.limit")) == ("8", "4")
    assert _flag(cmd, "--num-rollouts") == "3"
    assert _flag(cmd, "--output-dir") == str(Path("outputs/models/qwen3_5-9b"))
    assert _flag(cmd, "--run.dir") == "tasks-08-11"


def test_local_entry_gets_the_dummy_key_variable():
    entry = run_models.load_models()["qwen3-4b-vllm"]
    cmd = run_models.command("qwen3-4b-vllm", entry, _args(), first=0, n=4)
    assert _flag(cmd, "--client.api-key-var") == run_models.LOCAL_KEY_VAR


def test_vf_run_accepts_every_registry_entry(tmp_path):
    pytest.importorskip("fcntl")
    names = list(run_models.load_models())
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_models.py"), *names]
        + ["--secret", "x", "--dry-run", "--out", str(tmp_path), "--tasks", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    written = {p.parents[3].name for p in tmp_path.glob("*/*/configs/*/run.json")}
    assert written == set(names)


def test_chunk_is_done_only_when_every_episode_finished_ok(tmp_path):
    record = '{"ok": %s, "traces": []}\n'
    assert not run_models.chunk_done(tmp_path, 2)
    (tmp_path / "traces.jsonl").write_text(record % "true")
    assert not run_models.chunk_done(tmp_path, 2)
    (tmp_path / "traces.jsonl").write_text(record % "true" + record % "false")
    assert not run_models.chunk_done(tmp_path, 2)
    (tmp_path / "traces.jsonl").write_text(2 * (record % "true"))
    assert run_models.chunk_done(tmp_path, 2)


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
    row = results.episode_row("m", trace)
    assert (row["agent"], row["task"], row["reward"], row["ic"]) == ("m", 4, 0.3, 0.02)
    assert row["backtests_used"] == 3.0 and row["stop"] == "submitted"


def test_table_means_performance_over_valid_episodes_only():
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
    a = report.table(rows).row(0, named=True)
    assert a["Agent"] == "a" and a["Episodes"] == 2
    assert a["Net Sharpe"] == 2.0 and a["Invalid rate"] == 0.5
    assert math.isclose(a["Reward"], -0.25) and a["Backtests used"] == 3.5


def test_vf_config_file_is_passed_to_vf_run():
    entry = run_models.load_models()["stub"]
    args = _args(vf_config=Path("ic.toml"))
    cmd = run_models.command("stub", entry, args, first=0, n=1)
    assert cmd[-2:] == ["@", "ic.toml"]


def test_budget_sets_the_flag_and_names_the_run():
    entry = run_models.load_models()["qwen3_5-9b"]
    cmd = run_models.command("qwen3_5-9b", entry, _args(budget=3), first=0, n=4)
    assert _flag(cmd, "--env.taskset.task.tools.budget") == "3"
    assert _flag(cmd, "--run.name") == "qwen3_5-9b-budget3"


def test_claim_gap_pools_budgets_per_model():
    rows = pl.DataFrame(
        {
            "agent": ["m", "m-budget1", "m-budget1", "b"],
            "backtests_used": [10.0, 1.0, 1.0, 1.0],
            "claim_gap": [1.0, 0.2, float("nan"), 0.5],
        }
    )
    out = results.claim_gap_by_backtests(rows)
    assert out.get_column("model").to_list() == ["b", "m", "m"]
    m = out.filter(pl.col("model") == "m")
    assert m.get_column("backtests_used").to_list() == [1.0, 10.0]
    assert m.get_column("episodes").to_list() == [1, 1]
