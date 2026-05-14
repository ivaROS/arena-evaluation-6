from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


# ---------------------------------------------------------------------------
# Suite.parse
# ---------------------------------------------------------------------------

def _make_stage_dict(**overrides: object) -> dict:
    base: dict = {
        "name": "stage_one",
        "episodes": 5,
        "robot": "turtlebot3_burger",
        "map": "map1",
        "tm_robots": "RANDOM",
        "tm_obstacles": "RANDOM",
        "config": {},
    }
    base.update(overrides)
    return base


def test_suite_parse_happy_path():
    from arena_evaluation.benchmark.config import Suite

    obj = {"stages": [_make_stage_dict()]}
    result = Suite.parse("test_suite", obj)

    assert isinstance(result, Suite)
    assert result.name == "test_suite"
    assert len(result.stages) == 1
    assert isinstance(result.stages[0], Suite.Stage)


def test_suite_parse_multiple_stages():
    from arena_evaluation.benchmark.config import Suite

    obj = {"stages": [_make_stage_dict(name="s1"), _make_stage_dict(name="s2")]}
    result = Suite.parse("multi", obj)

    assert len(result.stages) == 2
    assert result.stages[0].name == "s1"
    assert result.stages[1].name == "s2"


def test_suite_parse_missing_stages_raises():
    from arena_evaluation.benchmark.config import Suite

    with pytest.raises(KeyError):
        Suite.parse("s", {})


# ---------------------------------------------------------------------------
# Suite.Stage.parse
# ---------------------------------------------------------------------------

def test_stage_parse_happy_path():
    from arena_evaluation.benchmark.config import Suite
    from task_generator.constants import Constants

    obj = _make_stage_dict(seed=42, timeout="60")
    result = Suite.Stage.parse(obj)

    assert isinstance(result, Suite.Stage)
    assert result.name == "stage_one"
    assert result.episodes == 5
    assert result.robot == "turtlebot3_burger"
    assert result.map == "map1"
    assert result.tm_robots is Constants.TaskMode.TM_Robots.RANDOM
    assert result.tm_obstacles is Constants.TaskMode.TM_Obstacles.RANDOM
    assert result.seed == 42
    assert result.timeout == 60.0


def test_stage_parse_missing_timeout_defaults_to_inf():
    import math

    from arena_evaluation.benchmark.config import Suite

    obj = _make_stage_dict()
    obj.pop("timeout", None)
    result = Suite.Stage.parse(obj)

    assert result.timeout == math.inf


def test_stage_parse_seed_defaults_to_hash():
    from arena_evaluation.benchmark.config import Suite

    obj = _make_stage_dict()
    obj.pop("seed", None)
    result = Suite.Stage.parse(obj)

    assert isinstance(result.seed, int)


def test_stage_parse_tm_robots_enum_conversion():
    from arena_evaluation.benchmark.config import Suite
    from task_generator.constants import Constants

    for name, member in Constants.TaskMode.TM_Robots.__members__.items():
        obj = _make_stage_dict(tm_robots=name)
        result = Suite.Stage.parse(obj)
        assert result.tm_robots is member


def test_stage_parse_tm_obstacles_enum_conversion():
    from arena_evaluation.benchmark.config import Suite
    from task_generator.constants import Constants

    for name, member in Constants.TaskMode.TM_Obstacles.__members__.items():
        obj = _make_stage_dict(tm_obstacles=name)
        result = Suite.Stage.parse(obj)
        assert result.tm_obstacles is member


def test_stage_parse_invalid_tm_robots_raises():
    from arena_evaluation.benchmark.config import Suite

    obj = _make_stage_dict(tm_robots="NOT_A_VALID_MODE")
    with pytest.raises(KeyError):
        Suite.Stage.parse(obj)


def test_stage_parse_invalid_tm_obstacles_raises():
    from arena_evaluation.benchmark.config import Suite

    obj = _make_stage_dict(tm_obstacles="NOT_A_VALID_MODE")
    with pytest.raises(KeyError):
        Suite.Stage.parse(obj)


# ---------------------------------------------------------------------------
# Contest.parse - list form
# ---------------------------------------------------------------------------

def test_contest_parse_list_happy_path():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "planner_a", "mobile.local_planner": "dwa"}]
    result = Contest.parse("my_contest", obj)

    assert isinstance(result, Contest)
    assert result.name == "my_contest"
    assert len(result.contestants) == 1
    c = result.contestants[0]
    assert c.name == "planner_a"
    assert c.args["mobile.local_planner"] == "dwa"


def test_contest_parse_list_multiple_contestants():
    from arena_evaluation.benchmark.config import Contest

    obj = [
        {"name": "p1", "mobile.local_planner": "dwa"},
        {"name": "p2", "mobile.local_planner": "teb"},
    ]
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 2
    assert result.contestants[0].name == "p1"
    assert result.contestants[1].name == "p2"


def test_contest_parse_list_description_is_none():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "p1", "mobile.local_planner": "dwa"}]
    result = Contest.parse("c", obj)

    assert result.description is None


def test_contest_parse_list_missing_name_raises():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"mobile.local_planner": "dwa"}]
    with pytest.raises(ValueError, match="requires 'name'"):
        Contest.parse("c", obj)


def test_contest_parse_list_duplicate_names_raises():
    from arena_evaluation.benchmark.config import Contest

    obj = [
        {"name": "same", "mobile.local_planner": "dwa"},
        {"name": "same", "mobile.local_planner": "teb"},
    ]
    with pytest.raises(ValueError, match="duplicate contestant name"):
        Contest.parse("c", obj)


def test_contest_parse_list_all_keys_except_name_go_to_args():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "p", "mobile.local_planner": "teb", "mobile.inter_planner": "foo", "mobile.agent": "bar"}]
    result = Contest.parse("c", obj)

    c = result.contestants[0]
    assert c.args == {"mobile.local_planner": "teb", "mobile.inter_planner": "foo", "mobile.agent": "bar"}


# ---------------------------------------------------------------------------
# Contest.parse - sweep form
# ---------------------------------------------------------------------------

def test_contest_parse_sweep_one_axis():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "teb"], "mobile.inter_planner": "foo"}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 2
    names = [c.name for c in result.contestants]
    assert names == ["dwa", "teb"]
    assert result.contestants[0].args["mobile.local_planner"] == "dwa"
    assert result.contestants[0].args["mobile.inter_planner"] == "foo"
    assert result.contestants[1].args["mobile.local_planner"] == "teb"
    assert result.contestants[1].args["mobile.inter_planner"] == "foo"


def test_contest_parse_sweep_two_axes():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "teb"], "mobile.global_planner": ["navfn", "smac"]}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 4
    names = [c.name for c in result.contestants]
    assert names == ["dwa-navfn", "dwa-smac", "teb-navfn", "teb-smac"]


def test_contest_parse_sweep_prefix():
    from arena_evaluation.benchmark.config import Contest

    obj = {"name": "foo", "mobile.local_planner": ["dwa", "teb"]}
    result = Contest.parse("c", obj)

    names = [c.name for c in result.contestants]
    assert names == ["foo-dwa", "foo-teb"]


def test_contest_parse_sweep_description_stored_not_in_args():
    from arena_evaluation.benchmark.config import Contest

    obj = {"description": "my desc", "mobile.local_planner": ["dwa", "teb"]}
    result = Contest.parse("c", obj)

    assert result.description == "my desc"
    for c in result.contestants:
        assert "description" not in c.args


def test_contest_parse_sweep_name_not_in_args():
    from arena_evaluation.benchmark.config import Contest

    obj = {"name": "prefix", "mobile.local_planner": ["dwa", "teb"]}
    result = Contest.parse("c", obj)

    for c in result.contestants:
        assert "name" not in c.args


def test_contest_parse_sweep_no_axes_single_contestant():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": "dwa", "mobile.inter_planner": "foo"}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 1
    assert result.contestants[0].name == "single"
    assert result.contestants[0].args["mobile.local_planner"] == "dwa"


def test_contest_parse_sweep_duplicate_names_raises():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "dwa"]}
    with pytest.raises(ValueError, match="duplicate contestant name"):
        Contest.parse("c", obj)


def test_contest_parse_sweep_constants_in_all_args():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "teb"], "mobile.inter_planner": "bypass"}
    result = Contest.parse("c", obj)

    for c in result.contestants:
        assert c.args["mobile.inter_planner"] == "bypass"


def test_contest_parse_invalid_type_raises():
    from arena_evaluation.benchmark.config import Contest

    with pytest.raises(ValueError, match="contest must be list or dict"):
        Contest.parse("c", "not_a_valid_type")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _parse_duration
# ---------------------------------------------------------------------------

def test_parse_duration_plain_int():
    from arena_evaluation.benchmark.config import _parse_duration
    assert _parse_duration("60") == 60.0


def test_parse_duration_plain_float():
    from arena_evaluation.benchmark.config import _parse_duration
    assert _parse_duration("60.0") == 60.0


def test_parse_duration_ms():
    from arena_evaluation.benchmark.config import _parse_duration
    assert _parse_duration("500ms") == pytest.approx(0.5)


def test_parse_duration_seconds_suffix():
    from arena_evaluation.benchmark.config import _parse_duration
    assert _parse_duration("5s") == pytest.approx(5.0)


def test_parse_duration_minutes():
    from arena_evaluation.benchmark.config import _parse_duration
    assert _parse_duration("5m") == pytest.approx(300.0)


def test_parse_duration_hours():
    from arena_evaluation.benchmark.config import _parse_duration
    assert _parse_duration("1h") == pytest.approx(3600.0)


def test_parse_duration_compound():
    from arena_evaluation.benchmark.config import _parse_duration
    assert _parse_duration("1h30m") == pytest.approx(5400.0)


def test_parse_duration_garbage_raises():
    from arena_evaluation.benchmark.config import _parse_duration
    with pytest.raises(ValueError):
        _parse_duration("not_a_duration")


def test_parse_duration_empty_raises():
    from arena_evaluation.benchmark.config import _parse_duration
    with pytest.raises(ValueError):
        _parse_duration("abc")
