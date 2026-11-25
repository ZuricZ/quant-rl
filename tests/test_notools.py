"""The no-tools taskset of the prime-rl smoke run, without a model or a
container."""

import asyncio
import dataclasses
import functools
import math

import pytest

# verifiers imports fcntl, which Windows lacks.
pytest.importorskip("fcntl")

import verifiers.v1 as vf  # noqa: E402
from test_taskset import SECRET, _LocalRuntime  # noqa: E402

from quant_research.servers.research import ResearchData  # noqa: E402
from quant_research.taskset import (  # noqa: E402
    SECRET_VAR,
    QuantResearchConfig,
    QuantResearchTaskset,
)
from quant_research_notools.taskset import (  # noqa: E402
    NoToolsConfig,
    NoToolsTask,
    NoToolsTaskConfig,
    NoToolsTaskset,
    extract_code,
    feature_table,
    ic_reward,
)
from quantenv import baselines, checks, metrics, reward, scoring  # noqa: E402
from quantenv.sources import synthetic  # noqa: E402

COLUMN = baselines.column_code("f00")
REPLY = f"I will use f00.\n\n```python\n{COLUMN}```\n"


def _trace(task: NoToolsTask, reply: str) -> vf.Trace:
    trace = vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="test", trainable=False),
    )
    trace.root_reply = reply
    return trace


@pytest.fixture(scope="module")
def task() -> NoToolsTask:
    return next(iter(NoToolsTaskset(NoToolsConfig(secret=SECRET))))


def _local_task() -> NoToolsTask:
    """Task 0 on the default market, graded as this machine's user, without
    a sandbox user."""
    config = NoToolsTaskConfig(secret=SECRET, sandbox_user=None, vary=False)
    return NoToolsTask(ResearchData(split="eval", index=0), config)


def test_splits_match_the_main_taskset():
    tasks = NoToolsTaskset(NoToolsConfig(secret=SECRET))
    main = QuantResearchTaskset(QuantResearchConfig(secret=SECRET))
    assert tasks.bounded
    assert [t.seed for t in tasks.take(3)] == [t.seed for t in main.take(3)]
    train = NoToolsTaskset(NoToolsConfig(secret=SECRET, split="train"))
    assert not train.bounded
    assert [t.data.name for t in train.take(2)] == ["train-0", "train-1"]


def test_secret_is_required(monkeypatch):
    monkeypatch.delenv(SECRET_VAR, raising=False)
    with pytest.raises(ValueError, match="secret"):
        NoToolsTaskset(NoToolsConfig())


def test_prompt_scores_every_feature_and_ret_on_visible_data(task):
    visible = synthetic.visible_data(task.seed, task.market)
    s = visible.panel.schema
    columns = [*s.features, s.ret]
    ics = metrics.mean_rank_ics(
        visible.panel.frame, {c: (c, s.target) for c in columns}, s.timestamp
    )
    prompt = task.data.prompt
    for c in columns:
        assert f"| {c} | {ics[c]:.4f} |" in prompt
    assert feature_table(visible) in prompt
    seed = task.seed
    for text in (str(seed), f"{seed:032x}", SECRET):
        assert text not in prompt


def test_prompt_states_the_answer_format_and_penalties(task):
    prompt = task.data.prompt
    assert "```python block" in prompt and "backtest" not in prompt
    assert f"divided by {checks.CheckConfig().max_ic:g}" in prompt
    assert "fades faster than" in prompt
    assert f"scores {reward.INVALID_REWARD:g}" in prompt
    assert f"future scores {reward.FLAGGED_REWARD:g}" in prompt


def test_extract_code_takes_the_last_python_block():
    reply = "```python\nfirst\n```\ntext\n```python\nsecond\n```"
    assert extract_code(reply) == "second\n"
    assert extract_code("```\nnot python\n```") is None
    assert extract_code("no code") is None


def test_extract_code_ignores_inline_reasoning():
    draft = "<think>\n```python\ndraft\n```\n</think>\n"
    assert extract_code(draft + "no code in the answer") is None
    assert extract_code(draft + "```python\nanswer\n```") == "answer\n"
    assert extract_code("<think>\n```python\ncut off\n```") is None


def test_reply_without_code_is_invalid(task):
    trace = _trace(task, "f00 looks best.")
    # Nothing to grade, so no runtime is needed.
    outcome = asyncio.run(task.outcome(trace, None))
    assert asyncio.run(task.hidden_ic(trace, None)) == reward.INVALID_REWARD
    assert outcome["no_code"] == 1 and outcome["invalid"] == 1
    assert math.isnan(outcome["raw_ic"])


def test_reward_is_the_hidden_ic_over_the_ceiling(tmp_path):
    task = _local_task()
    trace = _trace(task, REPLY)
    value = asyncio.run(task.hidden_ic(trace, _LocalRuntime(tmp_path)))
    hidden = functools.partial(synthetic.hidden_data, task.seed)
    expected = scoring.score_submission(COLUMN, 0.0, hidden)
    assert expected.invalid == 0 and expected.lookahead == 0
    max_ic = checks.CheckConfig().max_ic
    assert value == pytest.approx(expected.performance.ic / max_ic)
    # Graded once: the metric reuses the score kept on the trace.
    outcome = asyncio.run(task.outcome(trace, None))
    assert outcome["raw_ic"] == pytest.approx(expected.performance.ic)
    assert outcome["no_code"] == 0


def test_penalties_follow_the_main_reward_rule():
    performance = metrics.Performance(1000, 0.02, *[0.0] * 6, 3, *[0.0] * 4)
    honest = dataclasses.asdict(scoring.Score(performance, 0.0, 0, 0, 0, 0))
    assert ic_reward(honest, 0.05) == pytest.approx(0.4)
    assert ic_reward({**honest, "lookahead": 1}, 0.05) == reward.FLAGGED_REWARD
    assert ic_reward({**honest, "implausible_ic": 1}, 0.05) == reward.FLAGGED_REWARD
    invalid = dataclasses.asdict(scoring.invalid_score("boom"))
    assert ic_reward(invalid, 0.05) == reward.INVALID_REWARD


def test_episode_ends_after_one_reply(task):
    trace = _trace(task, REPLY)
    assert not task.answered(trace)
    trace.nodes.append(
        vf.MessageNode(message=vf.AssistantMessage(content=REPLY), sampled=True)
    )
    assert task.answered(trace)


def test_tasks_grade_in_a_fresh_container_of_the_image(task):
    assert task.NEEDS_CONTAINER
    assert task.scoring.mode == "isolated"
    assert task.data.image == "quant-research:0.1.0"
