"""The quant research taskset and toolset, without a model or a container."""

import asyncio
import dataclasses
import functools
import json

import pytest

# verifiers imports fcntl, which Windows lacks.
pytest.importorskip("fcntl")

import verifiers.v1 as vf  # noqa: E402

from quant_research import grade  # noqa: E402
from quant_research.servers.research import (  # noqa: E402
    ResearchConfig,
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
from quantenv import baselines, scoring  # noqa: E402
from quantenv.sources import synthetic  # noqa: E402

COLUMN = baselines.column_code("f00")


def _toolset(budget: int = 2) -> ResearchToolset:
    tools = ResearchToolset(ResearchConfig(budget=budget))
    asyncio.run(tools.setup_task(ResearchData(seed="0")))
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
def task() -> QuantResearchTask:
    return QuantResearchTask(ResearchData(seed="0"), QuantResearchTaskConfig())


def test_taskset_is_infinite_with_one_secret_seed_per_task():
    taskset = QuantResearchTaskset(QuantResearchConfig(secret="s"))
    assert taskset.INFINITE and not taskset.bounded
    tasks = list(taskset.take(2))
    assert [t.data.idx for t in tasks] == [0, 1]
    assert [t.data.dataset_seed for t in tasks] == [
        synthetic.dataset_seed(i, "s") for i in (0, 1)
    ]
    data = tasks[0].data
    assert data.seed not in data.prompt and str(data.dataset_seed) not in data.prompt


def test_secret_is_random_unless_set():
    assert QuantResearchConfig().secret != QuantResearchConfig().secret


def test_prompt_states_columns_budget_and_contract():
    config = QuantResearchConfig(
        task=QuantResearchTaskConfig(tools=ResearchConfig(budget=7))
    )
    prompt = next(iter(QuantResearchTaskset(config))).data.prompt
    assert "f00" in prompt and "ret" in prompt
    assert "You have 7 backtests" in prompt
    for name in (
        "features(df)",
        "fit(train)",
        "predict(df)",
        "submit(code, claimed_ic)",
    ):
        assert name in prompt


def test_describe_lists_visible_data_only():
    text = _toolset().describe()
    assert "fwd_return" not in text
    assert "timestamps: 1000" in text
    assert "assets: 200" in text


def test_backtest_matches_scoring_and_counts():
    tools = _toolset()
    out = asyncio.run(tools.backtest(COLUMN))
    ic = scoring.backtest(COLUMN, functools.partial(synthetic.visible_data, 0))
    assert f"rank IC: {ic:.4f}" in out
    assert "backtests left: 1" in out
    assert tools.state.backtests_used == 1


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
    assert state.submitted and state.code == COLUMN and state.claimed_ic == 0.05
    with pytest.raises(ValueError, match="refused"):
        tools.submit(COLUMN, 0.05)
    with pytest.raises(ValueError, match="refused"):
        asyncio.run(tools.backtest(COLUMN))
    assert tools.state.backtests_used == 0


def test_submit_stops_the_episode(task):
    assert not task.submitted(_trace(task, ResearchState()))
    assert task.submitted(_trace(task, ResearchState(submitted=True)))


def test_turn_limit_stops_the_episode():
    config = QuantResearchTaskConfig(max_turns=1)
    task = QuantResearchTask(ResearchData(seed="0"), config)
    trace = _trace(task, ResearchState())
    assert not task.turn_limit(trace)
    reply = vf.MessageNode(message=vf.AssistantMessage(content="hi"), sampled=True)
    trace.nodes.append(reply)
    assert task.turn_limit(trace)


def test_no_submission_is_invalid(task):
    trace = _trace(task, ResearchState(backtests_used=3))
    # Nothing to grade, so no runtime is needed.
    metrics = asyncio.run(task.outcome(trace, None))
    assert asyncio.run(task.true_ic(trace, None)) == 0.0
    assert metrics["invalid"] == 1
    assert metrics["backtests_used"] == 3


def test_grader_scores_on_hidden_periods(tmp_path):
    truth = synthetic.ground_truth(0)
    code = baselines.weighted_code(truth.weights)
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(
        json.dumps({"seed": 0, "code": code, "claimed_ic": 0.05, "user": None})
    )
    grade.main([str(request), str(result)])
    assert not request.exists()
    hidden = functools.partial(synthetic.hidden_data, 0)
    expected = scoring.score_submission(code, 0.05, hidden)
    score = json.loads(result.read_text())
    assert score["reward"] == pytest.approx(expected.reward)
    assert score["invalid"] == 0 and score["used_leak"] == 0
    assert set(score) == {f.name for f in dataclasses.fields(scoring.Score)}


def test_trace_of_a_real_task_serialises():
    # orjson rejects integers wider than 64 bits; dataset seeds have 128.
    task = next(iter(QuantResearchTaskset(QuantResearchConfig(secret="s"))))
    trace = _trace(task, ResearchState(submitted=True, code=COLUMN, claimed_ic=0.1))
    assert json.loads(trace.to_json())["task"]["data"]["name"] == "task-0"
