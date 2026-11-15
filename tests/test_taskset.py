"""The quant research taskset and toolset, without a model or a container."""

import asyncio
import dataclasses
import functools
import json

import pytest
from pydantic import ValidationError

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
SECRET = "a-test-secret-value"
DATA = ResearchData(split="eval", index=0)


def _config(**task) -> QuantResearchConfig:
    return QuantResearchConfig(secret=SECRET, task=QuantResearchTaskConfig(**task))


def _toolset(budget: int = 2) -> ResearchToolset:
    tools = ResearchToolset(ResearchConfig(budget=budget))
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
def task() -> QuantResearchTask:
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


def test_secret_is_required_and_masked():
    with pytest.raises(ValidationError):
        QuantResearchConfig()
    config = _config()
    assert SECRET not in config.model_dump_json()
    # A masked dump read back must fail loudly, not run on the mask.
    with pytest.raises(ValidationError):
        QuantResearchConfig.model_validate_json(config.model_dump_json())


def test_seed_stays_out_of_published_task_data(task):
    seed = synthetic.dataset_seed(0, SECRET, "eval")
    trace = _trace(task, ResearchState())
    asyncio.run(task.setup(trace, None))
    assert trace.state.seed == f"{seed:032x}"
    published = trace.to_json().decode()
    for text in (str(seed), f"{seed:032x}", SECRET):
        assert text not in published
        assert text not in task.data.prompt


def test_prompt_states_columns_budget_and_contract():
    config = _config(tools=ResearchConfig(budget=7))
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
    ic = scoring.backtest(
        COLUMN,
        functools.partial(
            synthetic.visible_data, synthetic.dataset_seed(0, SECRET, "eval")
        ),
    )
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
    task = QuantResearchTask(DATA, config)
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


def test_trace_of_a_real_task_serialises(task):
    # orjson rejects integers wider than 64 bits; dataset seeds have 128.
    trace = _trace(task, ResearchState(submitted=True, code=COLUMN, claimed_ic=0.1))
    asyncio.run(task.setup(trace, None))
    assert json.loads(trace.to_json())["task"]["data"]["name"] == "eval-0"
