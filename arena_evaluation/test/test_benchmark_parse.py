from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


# ---------------------------------------------------------------------------
# _Config.parse
# ---------------------------------------------------------------------------

def test_config_parse_happy_path():
    from task_generator.tasks.modules.benchmark.impl import _Config

    obj = {
        "suite": {"config": "my_suite.yaml", "scale_episodes": 2.5},
        "contest": {"config": "my_contest.yaml"},
        "general": {"simulator": "gazebo"},
    }
    result = _Config.parse(obj)

    assert isinstance(result, _Config)
    assert isinstance(result.suite, _Config.Suite)
    assert isinstance(result.contest, _Config.Contest)
    assert isinstance(result.general, _Config.General)
    assert result.suite.config == "my_suite.yaml"
    assert result.suite.scale_episodes == 2.5
    assert result.contest.config == "my_contest.yaml"
    assert result.general.simulator == "gazebo"


def test_config_parse_missing_suite_raises():
    from task_generator.tasks.modules.benchmark.impl import _Config

    obj = {
        "contest": {"config": "my_contest.yaml"},
        "general": {"simulator": "gazebo"},
    }
    with pytest.raises(KeyError):
        _Config.parse(obj)


def test_config_parse_missing_contest_raises():
    from task_generator.tasks.modules.benchmark.impl import _Config

    obj = {
        "suite": {"config": "my_suite.yaml"},
        "general": {"simulator": "gazebo"},
    }
    with pytest.raises(KeyError):
        _Config.parse(obj)


def test_config_parse_missing_general_raises():
    from task_generator.tasks.modules.benchmark.impl import _Config

    obj = {
        "suite": {"config": "my_suite.yaml"},
        "contest": {"config": "my_contest.yaml"},
    }
    with pytest.raises(KeyError):
        _Config.parse(obj)


def test_config_parse_suite_scale_episodes_defaults_to_one():
    from task_generator.tasks.modules.benchmark.impl import _Config

    obj = {
        "suite": {"config": "s.yaml"},
        "contest": {"config": "c.yaml"},
        "general": {"simulator": "flatland"},
    }
    result = _Config.parse(obj)
    assert result.suite.scale_episodes == 1


# ---------------------------------------------------------------------------
# Suite.parse
# ---------------------------------------------------------------------------

def _make_config_class_stub():
    """Return a minimal config_class stub that provides Robot.TIMEOUT."""
    class _Robot:
        TIMEOUT = 120

    class _Conf:
        Robot = _Robot

    return _Conf()


def _make_stage_dict(**overrides):
    base = {
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
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    obj = {"stages": [_make_stage_dict()]}
    result = Suite.parse("test_suite", obj, config_class)

    assert isinstance(result, Suite)
    assert result.name == "test_suite"
    assert len(result.stages) == 1
    assert isinstance(result.stages[0], Suite.Stage)


def test_suite_parse_multiple_stages():
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    obj = {"stages": [_make_stage_dict(name="s1"), _make_stage_dict(name="s2")]}
    result = Suite.parse("multi", obj, config_class)

    assert len(result.stages) == 2
    assert result.stages[0].name == "s1"
    assert result.stages[1].name == "s2"


def test_suite_parse_missing_stages_raises():
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    with pytest.raises(KeyError):
        Suite.parse("s", {}, config_class)


# ---------------------------------------------------------------------------
# Suite.Stage.parse
# ---------------------------------------------------------------------------

def test_stage_parse_happy_path():
    from task_generator.constants import Constants
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    obj = _make_stage_dict(seed=42, timeout="60")
    result = Suite.Stage.parse(obj, config_class)

    assert isinstance(result, Suite.Stage)
    assert result.name == "stage_one"
    assert result.episodes == 5
    assert result.robot == "turtlebot3_burger"
    assert result.map == "map1"
    assert result.tm_robots is Constants.TaskMode.TM_Robots.RANDOM
    assert result.tm_obstacles is Constants.TaskMode.TM_Obstacles.RANDOM
    assert result.seed == 42
    assert result.timeout == "60"


def test_stage_parse_none_config_class_raises():
    from task_generator.tasks.modules.benchmark.impl import Suite

    obj = _make_stage_dict()
    with pytest.raises(ValueError):
        Suite.Stage.parse(obj, None)


def test_stage_parse_timeout_defaults_from_config_class():
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    obj = _make_stage_dict()
    obj.pop("timeout", None)
    result = Suite.Stage.parse(obj, config_class)

    assert result.timeout == str(config_class.Robot.TIMEOUT)


def test_stage_parse_seed_defaults_to_hash():
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    obj = _make_stage_dict()
    obj.pop("seed", None)
    result = Suite.Stage.parse(obj, config_class)

    assert isinstance(result.seed, int)


def test_stage_parse_tm_robots_enum_conversion():
    from task_generator.constants import Constants
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    for name, member in Constants.TaskMode.TM_Robots.__members__.items():
        obj = _make_stage_dict(tm_robots=name)
        result = Suite.Stage.parse(obj, config_class)
        assert result.tm_robots is member


def test_stage_parse_tm_obstacles_enum_conversion():
    from task_generator.constants import Constants
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    for name, member in Constants.TaskMode.TM_Obstacles.__members__.items():
        obj = _make_stage_dict(tm_obstacles=name)
        result = Suite.Stage.parse(obj, config_class)
        assert result.tm_obstacles is member


def test_stage_parse_invalid_tm_robots_raises():
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    obj = _make_stage_dict(tm_robots="NOT_A_VALID_MODE")
    with pytest.raises(KeyError):
        Suite.Stage.parse(obj, config_class)


def test_stage_parse_invalid_tm_obstacles_raises():
    from task_generator.tasks.modules.benchmark.impl import Suite

    config_class = _make_config_class_stub()
    obj = _make_stage_dict(tm_obstacles="NOT_A_VALID_MODE")
    with pytest.raises(KeyError):
        Suite.Stage.parse(obj, config_class)


# ---------------------------------------------------------------------------
# Contest.parse
# ---------------------------------------------------------------------------

def _make_contestant_dict(**overrides):
    base = {"name": "planner_a", "local_planner": "dwa"}
    base.update(overrides)
    return base


def test_contest_parse_happy_path():
    from task_generator.tasks.modules.benchmark.impl import Contest

    obj = {"contestants": [_make_contestant_dict()]}
    result = Contest.parse("my_contest", obj)

    assert isinstance(result, Contest)
    assert result.name == "my_contest"
    assert len(result.contestants) == 1
    assert isinstance(result.contestants[0], Contest.Contestant)


def test_contest_parse_multiple_contestants():
    from task_generator.tasks.modules.benchmark.impl import Contest

    obj = {
        "contestants": [
            _make_contestant_dict(name="p1"),
            _make_contestant_dict(name="p2"),
        ]
    }
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 2
    assert result.contestants[0].name == "p1"
    assert result.contestants[1].name == "p2"


def test_contest_parse_missing_contestants_raises():
    from task_generator.tasks.modules.benchmark.impl import Contest

    with pytest.raises(KeyError):
        Contest.parse("c", {})


# ---------------------------------------------------------------------------
# Contest.Contestant.parse
# ---------------------------------------------------------------------------

def test_contestant_parse_happy_path():
    from task_generator.tasks.modules.benchmark.impl import Contest

    obj = {
        "name": "agent_one",
        "local_planner": "teb",
        "inter_planner": "custom_replanning",
        "agent_name": "rl_agent",
    }
    result = Contest.Contestant.parse(obj)

    assert isinstance(result, Contest.Contestant)
    assert result.name == "agent_one"
    assert result.local_planner == "teb"
    assert result.inter_planner == "custom_replanning"
    assert result.agent_name == "rl_agent"


def test_contestant_parse_inter_planner_defaults():
    from task_generator.tasks.modules.benchmark.impl import Contest

    obj = {"name": "agent_two", "local_planner": "dwa"}
    result = Contest.Contestant.parse(obj)

    assert result.inter_planner == "navigate_w_replanning_time"


def test_contestant_parse_agent_name_defaults_to_empty_string():
    from task_generator.tasks.modules.benchmark.impl import Contest

    obj = {"name": "agent_three", "local_planner": "dwa"}
    result = Contest.Contestant.parse(obj)

    assert result.agent_name == ""


# Contest.Contestant.parse uses only setdefault + cls(**obj), so a missing
# required NamedTuple field (name or local_planner) raises TypeError, not
# KeyError.  There is no explicit obj[...] access, so no KeyError test applies.
def test_contestant_parse_missing_name_raises():
    from task_generator.tasks.modules.benchmark.impl import Contest

    obj = {"local_planner": "dwa"}
    with pytest.raises(TypeError):
        Contest.Contestant.parse(obj)


def test_contestant_parse_missing_local_planner_raises():
    from task_generator.tasks.modules.benchmark.impl import Contest

    obj = {"name": "agent_four"}
    with pytest.raises(TypeError):
        Contest.Contestant.parse(obj)
