"""The quant research taskset and toolset, without a model or a container."""

import asyncio
import dataclasses
import functools
import json
import math
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

# verifiers imports fcntl, which Windows lacks.
pytest.importorskip("fcntl")

import verifiers.v1 as vf  # noqa: E402
from conftest import SMALL_MARKET  # noqa: E402

from quant_research import grade, taskset  # noqa: E402
from quant_research.servers.research import (  # noqa: E402
    ResearchConfig,
    backtest_report,
    backtest_scope,
    ResearchData,
    ResearchState,
    ResearchToolset,
)
from quant_research.taskset import (  # noqa: E402
    QuantResearchConfig,
    QuantResearchTask,
    QuantResearchTaskConfig,
    QuantResearchTaskset,
)
from quantenv import baselines, checks, metrics, reward, scoring, walkforward  # noqa: E402
from quantenv.alpha import AlphaError  # noqa: E402
from quantenv.sources import synthetic  # noqa: E402

COLUMN = baselines.column_code("f00")
SECRET = "a-test-secret-value"

# Every task here gets a small market (see conftest.py).
pytestmark = pytest.mark.usefixtures("small_market")
DATA = ResearchData(split="eval", index=0)


# Agent code runs as this process's user: switching to the sandbox user needs
# root, and tests/test_sandbox.py covers that path.
TOOLS = ResearchConfig(sandbox_user=None)


def _config(**task) -> QuantResearchConfig:
    task.setdefault("tools", TOOLS)
    return QuantResearchConfig(secret=SECRET, task=QuantResearchTaskConfig(**task))


def _toolset(budget: int = 2) -> ResearchToolset:
    tools = ResearchToolset(ResearchConfig(budget=budget, sandbox_user=None))
    asyncio.run(tools.setup_task(DATA))
    # Outside a call the toolset sees its inert state; the host's Task.setup
    # puts the seed there in a rollout.
    tools.state.seed = f"{synthetic.dataset_seed(0, SECRET, 'eval'):032x}"
    return tools


def _trace(task: QuantResearchTask, state: ResearchState) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        state=state,
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="test", trainable=False),
    )


@pytest.fixture(scope="module")
def task(small_market) -> QuantResearchTask:
    return next(iter(QuantResearchTaskset(_config())))


def test_train_and_eval_seeds_are_disjoint():
    seeds = {
        split: {synthetic.dataset_seed(i, SECRET, split) for i in range(10_000)}
        for split in ("train", "eval")
    }
    assert len(seeds["train"]) == len(seeds["eval"]) == 10_000
    assert not seeds["train"] & seeds["eval"]


def test_eval_split_is_30_named_tasks_and_train_is_infinite():
    eval_set = QuantResearchTaskset(_config())
    assert eval_set.bounded
    tasks = list(eval_set)
    assert len(tasks) == 30
    assert tasks[0].data.name == "eval-0" and tasks[-1].data.name == "eval-29"
    train = QuantResearchTaskset(QuantResearchConfig(secret=SECRET, split="train"))
    assert not train.bounded
    assert [t.data.name for t in train.take(2)] == ["train-0", "train-1"]


def test_secret_is_required_and_masked(monkeypatch):
    monkeypatch.delenv(taskset.SECRET_VAR, raising=False)
    # verifiers' CLI builds the config with defaults before reading flags, so
    # the field may be unset; the taskset refuses to build without it.
    config = QuantResearchConfig()
    with pytest.raises(ValueError, match="secret"):
        QuantResearchTaskset(config)
    config = _config()
    assert SECRET not in config.model_dump_json()
    # A masked dump read back must fail loudly, not run on the mask.
    with pytest.raises(ValidationError):
        QuantResearchConfig.model_validate_json(config.model_dump_json())


def test_secret_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv(taskset.SECRET_VAR, SECRET)
    assert QuantResearchConfig().secret.get_secret_value() == SECRET
    # A secret that is set wins over the environment.
    assert QuantResearchConfig(secret="other").secret.get_secret_value() == "other"
    monkeypatch.setenv(taskset.SECRET_VAR, "")
    assert QuantResearchConfig().secret is None


def test_seed_stays_out_of_published_task_data(task, tmp_path):
    seed = synthetic.dataset_seed(0, SECRET, "eval")
    trace = _trace(task, ResearchState())
    asyncio.run(task.setup(trace, _LocalRuntime(tmp_path)))
    assert trace.state.seed == f"{seed:032x}"
    published = trace.to_json().decode()
    for text in (str(seed), f"{seed:032x}", SECRET):
        assert text not in published
        assert text not in task.data.prompt


def test_prompt_states_columns_budget_and_contract():
    config = _config(tools=ResearchConfig(budget=7, sandbox_user=None))
    task = next(iter(QuantResearchTaskset(config)))
    prompt = task.data.prompt
    assert task.market.feature_names[0] in prompt and "ret" in prompt
    assert "You have 7 backtests" in prompt
    for name in (
        "features(df)",
        "fit(train)",
        "predict(df)",
        "submit(code, claimed_sharpe)",
    ):
        assert name in prompt


def test_prompt_offers_the_seeded_generator():
    prompt = next(iter(QuantResearchTaskset(_config()))).data.prompt
    assert "self.rng" in prompt


# Words that would describe a check or the reward; neither belongs in a prompt.
LEAK_WORDS = (
    "ceiling",
    "flag",
    "caught",
    "penal",
    "reward",
    "lookahead",
    "implausible",
    "decay",
    "fades",
    "scores -",
)


def test_prompt_names_the_libraries_pinned_in_pyproject():
    prompt = next(iter(QuantResearchTaskset(_config()))).data.prompt
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    deps = project["project"]["dependencies"]
    pins = dict(dep.split("==") for dep in deps if "==" in dep)
    for name in ("polars", "polars-ds", "scikit-learn", "scipy", "numpy"):
        assert f"{name} {pins[name]}" in prompt
    assert "nothing else" in prompt


def test_prompt_states_no_checks_or_reward():
    prompt = next(iter(QuantResearchTaskset(_config()))).data.prompt
    max_ic = QuantResearchTaskConfig().check_config.max_ic
    assert f"IC of {max_ic:g}" not in prompt
    for word in LEAK_WORDS:
        assert word not in prompt.lower()
    assert "only a submitted alpha counts" in prompt


def test_prompt_states_the_hidden_run_and_its_time_limit():
    task = next(iter(QuantResearchTaskset(_config())))
    prompt = task.data.prompt
    assert f"{task.market.n_hidden:,} unseen periods" in prompt
    limit = scoring.TimeLimit()
    seconds = limit.seconds(taskset.run_rows(hidden=True, config=task.market))
    assert f"so it may take {seconds:.1f} seconds: an alpha" in prompt
    seconds = limit.seconds(taskset.run_rows(hidden=False, config=task.market))
    assert f"so it may take {seconds:.1f} seconds." in prompt


def test_prompt_states_the_markets_limits():
    prompt = next(iter(QuantResearchTaskset(_config(vary=False)))).data.prompt
    assert f"{SMALL_MARKET.n_hidden:,} unseen periods" in prompt
    seconds = scoring.TimeLimit().seconds(taskset.run_rows(False, SMALL_MARKET))
    assert f"so it may take {seconds:.1f} seconds." in prompt


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("variation", [None, synthetic.variation_seed(3, "eval", 0)])
def test_prompt_rows_match_the_data(hidden, variation):
    seed = synthetic.dataset_seed(0, "x", "eval")
    market = synthetic.task_config(variation)
    load = synthetic.hidden_data if hidden else synthetic.visible_data
    data = load(seed, market)
    assert taskset.run_rows(hidden, market) == walkforward.run_rows(
        data.panel, walkforward.WalkForward(), data.start
    )


def test_unvaried_prompt_keeps_the_fixed_opening_and_example():
    task = next(iter(QuantResearchTaskset(_config(vary=False))))
    prompt = task.data.prompt
    assert task.variation is None and task.market == SMALL_MARKET
    assert prompt.startswith(taskset.FIXED_OPENING)
    assert taskset.EXAMPLE in prompt and "## Request" not in prompt


def test_varied_prompts_state_the_requested_style():
    tasks = list(QuantResearchTaskset(_config()).take(5))
    # Eval task i asks for the i-th style.
    assert [t.market.style for t in tasks] == list(synthetic.STYLES)
    for task in tasks:
        prompt = task.data.prompt
        request = taskset.STYLE_REQUESTS[task.market.style]
        if request:
            assert f"## Request\n\n{request}" in prompt
        else:
            assert "## Request" not in prompt
        assert prompt.split("\n\n")[0] in taskset.OPENINGS
        assert taskset.WORKFLOW.rstrip() in prompt
        assert taskset.PATTERN_EXAMPLE in prompt
        assert taskset.EXAMPLE not in prompt  # Never the ready-made example.


def test_pattern_example_runs_once_its_placeholder_is_a_column():
    code = taskset.PATTERN_EXAMPLE.split("```python\n")[1].split("```")[0]
    load = functools.partial(synthetic.visible_data, 0, SMALL_MARKET)
    # As it stands it cannot be submitted: "COLUMN" is not a column.
    assert "COLUMN" not in load().panel.frame.columns
    with pytest.raises(AlphaError, match="COLUMN"):
        scoring.backtest(code, load)
    # With a real column it is a valid alpha, so the pattern is right.
    performance = scoring.backtest(code.replace("COLUMN", "f00"), load)
    assert performance.periods > 0


def test_toolset_backtests_the_varied_market():
    tools = _toolset()
    variation = synthetic.variation_seed(1, "eval", 0)
    tools.state.variation = f"{variation:032x}"
    market = synthetic.task_config(variation)
    text = asyncio.run(tools.describe())
    assert market.feature_names[0] in text
    assert f"assets: {market.n_assets}" in text


def test_grading_timeout_covers_every_scored_run():
    config = checks.CheckConfig()
    seconds = scoring.TimeLimit().seconds(taskset.run_rows(hidden=True))
    assert taskset.scoring_timeout(config) > (2 + config.n_cuts) * seconds


def test_describe_lists_visible_data_only():
    text = asyncio.run(_toolset().describe())
    assert "fwd_return" not in text
    assert f"timestamps: {SMALL_MARKET.n_visible}" in text
    assert f"assets: {SMALL_MARKET.n_assets}" in text


def test_backtest_matches_scoring_and_counts():
    tools = _toolset()
    out = asyncio.run(tools.backtest(COLUMN))
    performance = scoring.backtest(
        COLUMN,
        functools.partial(
            synthetic.visible_data, synthetic.dataset_seed(0, SECRET, "eval"), SMALL_MARKET
        ),
    )
    assert out == backtest_report(
        backtest_scope(
            synthetic.visible_data(synthetic.dataset_seed(0, SECRET, "eval"), SMALL_MARKET)
        ),
        performance,
        left=1,
    )
    assert tools.state.backtests_used == 1


def test_backtest_report_explains_its_numbers():
    performance = metrics.Performance(
        periods=1000,
        ic=0.012,
        ic_se=0.003,
        sharpe=0.85,
        sharpe_se=0.5,
        robust_sharpe=0.8,
        gross_sharpe=1.6,
        turnover=0.43,
        years=3,
        year_sharpe_min=-0.8,
        year_sharpe_median=0.9,
        year_sharpe_max=2.1,
        max_drawdown=0.1,
    )
    scope = backtest_scope(synthetic.visible_data(0, SMALL_MARKET))
    n = SMALL_MARKET.n_visible
    window = walkforward.WalkForward().window
    assert f"last {n - window} of {n} visible timestamps" in scope
    assert "252 periods per year" in scope and "2 bp" in scope
    out = backtest_report(scope, performance, left=4)
    for text in (
        "net Sharpe: 0.85 ± 0.50",
        "gross Sharpe: 1.60",
        "turnover: 0.43",
        "rank IC: 0.0120 ± 0.0030",
        "1000 periods",
        "min -0.80, median 0.90, max 2.10",
        "backtests left: 4",
    ):
        assert text in out
    # Numbers only: no advice that would coach the claim.
    for word in ("should", "try", "claim"):
        assert word not in out.lower()


def _registered(config: ResearchConfig) -> set[str]:
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("test")
    ResearchToolset(config).register(mcp)
    return {tool.name for tool in asyncio.run(mcp.list_tools())}


def test_optional_tools_are_off_by_default():
    assert _registered(TOOLS) == {"describe", "backtest", "submit"}
    prompt = next(iter(QuantResearchTaskset(_config()))).data.prompt
    assert "check(code)" not in prompt


def test_check_is_offered_when_turned_on():
    tools = ResearchConfig(sandbox_user=None, check=True)
    assert _registered(tools) == {"describe", "check", "backtest", "submit"}
    prompt = next(iter(QuantResearchTaskset(_config(tools=tools)))).data.prompt
    assert taskset.OPTIONAL_TOOL_LINES["check"] in prompt


def test_check_runs_the_code_without_using_the_budget():
    tools = _toolset(budget=1)
    assert asyncio.run(tools.check(COLUMN)).startswith("ok")
    out = asyncio.run(tools.check(baselines.column_code("no_such_column")))
    assert out.startswith("error:") and "ColumnNotFoundError" in out
    assert tools.state.backtests_used == 0
    # Numbers only come from backtests.
    assert not any(ch.isdigit() for ch in asyncio.run(tools.check(COLUMN)))


def test_check_refused_after_submit():
    tools = _toolset()
    tools.submit(COLUMN, 0.05)
    with pytest.raises(ValueError, match="refused"):
        asyncio.run(tools.check(COLUMN))


def test_failing_backtest_counts_against_budget():
    tools = _toolset()
    out = asyncio.run(tools.backtest("not python"))
    assert out.startswith("error:")
    assert tools.state.backtests_used == 1


def test_backtest_refused_past_budget():
    tools = _toolset(budget=1)
    asyncio.run(tools.backtest(COLUMN))
    with pytest.raises(ValueError, match="refused"):
        asyncio.run(tools.backtest(COLUMN))
    assert tools.state.backtests_used == 1


def test_submit_records_and_refuses_further_calls():
    tools = _toolset()
    tools.submit(COLUMN, 0.05)
    state = tools.state
    assert state.submitted and state.code == COLUMN and state.claimed_sharpe == 0.05
    with pytest.raises(ValueError, match="refused"):
        tools.submit(COLUMN, 0.05)
    with pytest.raises(ValueError, match="refused"):
        asyncio.run(tools.backtest(COLUMN))
    assert tools.state.backtests_used == 0


def test_submit_stops_the_episode(task):
    assert not task.submitted(_trace(task, ResearchState()))
    assert task.submitted(_trace(task, ResearchState(submitted=True)))


def test_turn_limit_stops_the_episode_as_truncated():
    config = QuantResearchTaskConfig(max_turns=1)
    task = QuantResearchTask(DATA, config)
    trace = _trace(task, ResearchState())
    assert not task.max_turns(trace)
    reply = vf.MessageNode(message=vf.AssistantMessage(content="hi"), sampled=True)
    trace.nodes.append(reply)
    assert task.max_turns(trace)
    # verifiers names the stop after the hook; "max_turns" counts as a
    # truncation, which scoring and training treat as such.
    trace.stop(task.max_turns.__name__)
    assert trace.is_truncated


def test_no_submission_is_invalid(task):
    trace = _trace(task, ResearchState(backtests_used=3))
    # Nothing to grade, so no runtime is needed.
    metrics = asyncio.run(task.outcome(trace, None))
    assert asyncio.run(task.hidden_sharpe(trace, None)) == reward.INVALID_REWARD
    assert metrics["invalid"] == 1
    assert metrics["backtests_used"] == 3
    assert math.isnan(metrics["sharpe"]) and math.isnan(metrics["claim_gap"])


def test_grader_scores_on_hidden_periods(tmp_path):
    truth = synthetic.ground_truth(0, SMALL_MARKET)
    code = baselines.weighted_code(truth.weights)
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(
        json.dumps(
            {
                "seed": 0,
                "code": code,
                "claimed_sharpe": 1.0,
                "user": None,
                "checks": {},
                "variation": None,
            }
        )
    )
    grade.main([str(request), str(result)])
    assert not request.exists()
    hidden = functools.partial(synthetic.hidden_data, 0, SMALL_MARKET)
    expected = scoring.score_submission(code, 1.0, hidden)
    score = json.loads(result.read_text())
    assert "reward" not in score  # The host applies the rule, not the image.
    assert reward.reward(score) == pytest.approx(
        reward.reward(dataclasses.asdict(expected))
    )
    assert score["performance"]["sharpe"] == pytest.approx(expected.performance.sharpe)
    assert score["invalid"] == 0 and score["used_leak"] == 0
    assert set(score) == {f.name for f in dataclasses.fields(scoring.Score)}


def test_trace_of_a_real_task_serialises(task, tmp_path):
    # orjson rejects integers wider than 64 bits; dataset seeds have 128.
    trace = _trace(task, ResearchState(submitted=True, code=COLUMN, claimed_sharpe=0.1))
    asyncio.run(task.setup(trace, _LocalRuntime(tmp_path)))
    assert json.loads(trace.to_json())["task"]["data"]["name"] == "eval-0"


class _LocalRuntime:
    """The parts of vf.Runtime that the tasks use, run on this machine, as the
    task image would: `python` is an interpreter with this package. The
    grader runs in this process, so a market patched by a fixture
    (`small_market`) reaches it."""

    def __init__(self, tmp_path):
        self.argv: list[str] = []
        self._bin = tmp_path / "bin"
        self._bin.mkdir()
        python = self._bin / "python"
        python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
        python.chmod(0o755)

    async def write(self, path: str, data: bytes) -> None:
        Path(path).write_bytes(data)

    async def run(self, argv: list[str], env: dict[str, str]):
        self.argv = argv
        if argv[:3] == ["python", "-m", "quant_research.grade"]:
            grade.main(argv[3:])
            return vf.ProgramResult(exit_code=0, stdout="", stderr="")
        path = f"{self._bin}:{os.environ['PATH']}"
        done = subprocess.run(
            argv, env={**os.environ, "PATH": path}, capture_output=True, text=True
        )
        return vf.ProgramResult(
            exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    async def read(self, path: str) -> bytes:
        return Path(path).read_bytes()


def test_grade_runs_the_images_python(tmp_path):
    runtime = _LocalRuntime(tmp_path)
    seed = synthetic.dataset_seed(0, SECRET, "eval")
    score = asyncio.run(taskset.grade(None, runtime, seed, COLUMN, 0.5))
    assert runtime.argv[:3] == ["python", "-m", "quant_research.grade"]
    assert score["invalid"] == 0 and score["performance"]["sharpe"] is not None


def test_tasks_use_the_configured_image():
    task = next(iter(QuantResearchTaskset(_config())))
    assert task.data.image == taskset.IMAGE == "quant-research:0.1.3"
    other = QuantResearchConfig(
        secret=SECRET, task=QuantResearchTaskConfig(image="registry.example/qr:1")
    )
    assert next(iter(QuantResearchTaskset(other))).data.image == "registry.example/qr:1"


def test_scoring_runs_in_a_fresh_container():
    # The grader must not run where agent code ran during the episode.
    task = next(iter(QuantResearchTaskset(_config())))
    assert task.scoring.mode == "isolated"


def test_setup_refuses_an_unusable_sandbox_user(tmp_path):
    config = QuantResearchTaskConfig(
        secret=SECRET, tools=ResearchConfig(sandbox_user="no-such-user-xyz")
    )
    task = QuantResearchTask(DATA, config)
    with pytest.raises(RuntimeError, match="sandbox user"):
        asyncio.run(task.setup(_trace(task, ResearchState()), _LocalRuntime(tmp_path)))


async def plugged_reward(task, trace, runtime) -> float:
    """A stand-in for a reward plugged by config in place of hidden_sharpe."""
    score = await task.raw_score(trace, runtime)
    return 7.0 if score["invalid"] else 0.0


def test_a_plugged_reward_replaces_hidden_sharpe_and_gets_the_raw_score():
    config = QuantResearchTaskConfig(
        secret=SECRET,
        rewards={"hidden_sharpe": {"fn": "test_taskset:plugged_reward"}},
    )
    task = QuantResearchTask(DATA, config)
    (fn,) = task.hooks("reward")
    assert fn.__name__ == "hidden_sharpe"
    trace = _trace(task, ResearchState())  # No submission: invalid.
    assert asyncio.run(fn(task=task, trace=trace, runtime=None)) == 7.0


def test_preflight_module_rejects_an_unknown_user():
    run = subprocess.run(
        [sys.executable, "-m", "quant_research.preflight", "no-such-user-xyz"],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0 and "no-such-user-xyz" in run.stderr
