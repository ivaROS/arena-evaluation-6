from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import logging
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import types
import typing

_T = typing.TypeVar("_T")

import attrs
from arena_robots_msgs.msg import CollisionEvents
from nav2_msgs.msg import CollisionMonitorState
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_evaluation_msgs.msg import BenchmarkState
from arena_rclpy_mixins import ActionClientWrapper, ArenaMixinNode, ClientWrapper
from arena_runtime_msgs.msg import EnvRecord, EnvRegistry
from arena_runtime_msgs.srv import DespawnEnv, SpawnEnv
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from task_generator.constants import Constants
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.msg import EpisodeRecord, RobotFleet
from task_generator_msgs.srv import QueueEpisode

STATE_TOPIC = "/arena/benchmark/state"

from .config import Contest, Suite
from .state import (
    Manifest,
    RunDir,
    capture_git_sha,
    compute_config_hash,
    find_most_recent_resumable,
)
from .step import Step, StepErrorKind, StepResult

# ---------------------------------------------------------------------------
# Free functions (testable without ROS init)
# ---------------------------------------------------------------------------


class _WithSteps(typing.Protocol):
    steps: dict[str, StepResult]


class _HasStateSteps(typing.Protocol):
    """Structural interface required by build_pending: any object with .state.steps."""

    @property
    def state(self) -> _WithSteps: ...


_log = logging.getLogger(__name__)


def build_launch_args(step: Step, simulator: str | None) -> list[str]:
    """Return the arena launch argument list for a step, given the simulator name.

    Per-mode params (task.scenario.file, task.random.*, ...) are not passed here;
    the runner sets them via QueueEpisode before each RunEpisode goal.

    Contestant args are forwarded verbatim, except keys that collide with
    stage-owned launch args (sim, robot, world, tm_robots, tm_obstacles,
    run_seed, auto_reset, tm_modules, record_data_dir), which would override
    the stage's configuration. Those are logged and dropped.
    """
    s = step.stage
    args = [
        *([f"sim:={simulator}"] if simulator is not None else []),
        f"robot:={s.robot}",
        f"world:={s.map}",
        f"tm_robots:={s.tm_robots.value}",
        f"tm_obstacles:={s.tm_obstacles.value}",
        f"run_seed:={s.seed}",
        "auto_reset:=false",
        "tm_modules:=",
    ]
    if step.record_dir is not None:
        args.append(f"record_data_dir:={step.record_dir}")
    own_keys = {a.split(":=", 1)[0] for a in args}
    for k, v in step.contestant.args.items():
        if v is None or v == "":
            continue
        if k in own_keys:
            _log.warning(
                "contestant %r: arg %r=%r ignored, controlled by stage",
                step.contestant.name, k, v,
            )
            continue
        args.append(f"{k}:={v}")
    return args


def make_timeout_episode_record(step: Step, episode_id: int):
    obs_params, rob_params = _flatten_per_mode_params(
        step.stage.config,
        tm_obstacles=step.stage.tm_obstacles.value,
        tm_robots=step.stage.tm_robots.value,
    )
    return types.SimpleNamespace(
        episode_id=episode_id,
        world=step.stage.map,
        seed=step.stage.seed,
        tm_robots=step.stage.tm_robots.value,
        tm_obstacles=step.stage.tm_obstacles.value,
        tm_modules=[],
        robots=[name for name in str(step.stage.robot).split(",") if name],
        outcome_state=EpisodeRecord.FAILED,
        outcome_info="timeout",
        robots_params=rob_params,
        obstacles_params=obs_params,
    )


def build_pending(
    suite: Suite,
    contest: Contest,
    scale_episodes: float,
    run_dir: _HasStateSteps,
    retry_failed: bool,
    record_root: pathlib.Path,
) -> list[Step]:
    """Return the list of steps that still need to be run.

    Retry policy:
      - Not in state file   -> run.
      - status: ok          -> skip (done).
      - status: failed      -> skip unless retry_failed=True.
      - status: partial     -> always retry; partial steps are definitionally
                               incomplete (some episodes failed or the run was
                               interrupted), so they need a full re-run.
      - status: skipped     -> run again (skipped = cancelled, deserves a fresh try).
      - status: in_progress -> run again (interrupted mid-flight).
    """
    state_steps = run_dir.state.steps
    steps: list[Step] = []
    seen: set[str] = set()
    for contestant in contest.contestants:
        for stage in suite.stages:
            step = Step(
                contestant=contestant,
                stage=stage,
                episodes=int(round(stage.episodes * scale_episodes)),
                record_dir=record_root / contestant.name / stage.name,
            )
            if step.key in seen:
                raise ValueError(f"duplicate step key: {step.key!r}")
            seen.add(step.key)
            existing = state_steps.get(step.key)
            if existing is None:
                steps.append(step)
                continue
            if existing.status == "ok":
                continue
            if existing.status == "failed" and not retry_failed:
                continue
            # partial, skipped, in_progress, or failed+retry_failed: (re-)run.
            steps.append(step)
    return steps


def env_key(step: Step, simulator: str | None) -> tuple:
    """Steps with the same env_key reuse one env. Contestants always force a new env."""
    # The recorder opens its rosbag when the env is spawned. Include the
    # per-step record directory in the key so each documented
    # <contestant>/<stage>/recording path gets its own writer.
    return (step.contestant.name, step.stage.robot, simulator, step.record_dir)


def group_pending(pending: list[Step], simulator: str | None) -> list[list[Step]]:
    """Group consecutive steps with the same env_key, preserving suite order.

    Splitting only happens when env_key changes between adjacent steps.
    """
    if not pending:
        return []
    groups: list[list[Step]] = []
    current: list[Step] = [pending[0]]
    current_key = env_key(pending[0], simulator)
    for step in pending[1:]:
        k = env_key(step, simulator)
        if k == current_key:
            current.append(step)
        else:
            groups.append(current)
            current = [step]
            current_key = k
    groups.append(current)
    return groups


def _walk_dict(d: dict, prefix: str = "") -> list[Parameter]:
    """Flatten a nested dict to rcl_interfaces Parameter[] with dot-joined leaf names."""
    out: list[Parameter] = []
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_walk_dict(v, name))
            continue
        pv = ParameterValue()
        if isinstance(v, bool):
            pv.type = ParameterType.PARAMETER_BOOL
            pv.bool_value = v
        elif isinstance(v, int):
            pv.type = ParameterType.PARAMETER_INTEGER
            pv.integer_value = v
        elif isinstance(v, float):
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = v
        elif isinstance(v, str):
            pv.type = ParameterType.PARAMETER_STRING
            pv.string_value = v
        else:
            raise TypeError(f"unsupported param type for {name!r}: {type(v).__name__}")
        p = Parameter()
        p.name = name
        p.value = pv
        out.append(p)
    return out


def _flatten_per_mode_params(
    stage_config: dict,
    *,
    tm_obstacles: str,
    tm_robots: str,
) -> tuple[list[Parameter], list[Parameter]]:
    """Route stage.config blocks to (obstacles_params, robots_params) as leaf-keyed Parameter[].

    Top-level keys in stage_config are mode names matching tm_obstacles / tm_robots
    (e.g. ``scenario``, ``random``). Each block is flattened to leaves relative to the
    mode (e.g. ``static.n``, ``file``) per QueueEpisode contract. ``scenario.file`` values
    are stripped to the stem (no path/extension).
    """
    obs: list[Parameter] = []
    rob: list[Parameter] = []
    for mode, mode_dict in (stage_config or {}).items():
        if not isinstance(mode_dict, dict):
            continue
        is_scenario = mode == "scenario"
        patched: dict = {
            k: (pathlib.Path(val).stem if is_scenario and k == "file" and isinstance(val, str) else val)
            for k, val in mode_dict.items()
        }
        params = _walk_dict(patched)
        if mode == tm_obstacles:
            obs.extend(params)
        if mode == tm_robots:
            rob.extend(params)
    return obs, rob


_LATCHED = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# Failures that mean the env itself is unusable, so the rest of the group
# (and possibly the run) cannot proceed. Episode-level failures (no record,
# timeout, robot got stuck, etc.) are NOT systemic, the next step can still run.
_SYSTEMIC = (StepErrorKind.ENV_SETUP, StepErrorKind.ROBOT_SETUP)


class _EnvDied(Exception):
    """Raised when an env disappears from /arena/state/envs while the runner was waiting on it."""


class CollisionAccumulator:
    """Counts new collision contacts while one benchmark episode is active."""

    _EVENT_SOURCE = "collision_events"
    _STATE_SOURCE = "collision_monitor_state"
    _STOP_ACTION = 1

    def __init__(self) -> None:
        self._active = False
        self._current_by_robot_source: dict[tuple[str, str], set[tuple[str, str]]] = {}
        self._event_contacts_by_robot: dict[str, dict[tuple[str, str], set[str]]] = {}
        self._active_event_by_incident: dict[tuple[str, str], int] = {}
        self._robots_with_events: set[str] = set()
        self._events: list[dict[str, object]] = []

    def begin(self) -> None:
        self._active = True
        self._current_by_robot_source.clear()
        self._event_contacts_by_robot.clear()
        self._active_event_by_incident.clear()
        self._robots_with_events.clear()
        self._events = []

    def end(self) -> tuple[int, list[dict[str, object]]]:
        self._active = False
        self._current_by_robot_source.clear()
        self._event_contacts_by_robot.clear()
        self._active_event_by_incident.clear()
        self._robots_with_events.clear()
        return len(self._events), list(self._events)

    def on_events(self, robot_ns: str, msg: CollisionEvents) -> None:
        if not self._active:
            return
        robot_name = self._robot_name(robot_ns)
        if robot_ns not in self._robots_with_events:
            self._drop_state_fallback(robot_ns)
            self._robots_with_events.add(robot_ns)
        previous_global = self._global_event_contacts()
        current: dict[tuple[str, str], set[str]] = {}
        current_robot_ns_by_incident: dict[tuple[str, str], str] = {}
        for event in msg.events:
            if not event.polygon_name and not event.obstacle_id:
                continue
            obstacle_id = self._canonical_obstacle_id(robot_name, event.obstacle_id)
            incident = (self._collision_owner(robot_name, event.obstacle_id), obstacle_id)
            current.setdefault(incident, set()).add(event.polygon_name)
            current_robot_ns_by_incident[incident] = robot_ns
        self._event_contacts_by_robot[robot_ns] = current
        current_global = self._global_event_contacts()
        for incident in sorted(set(previous_global) - set(current_global)):
            self._active_event_by_incident.pop(incident, None)
        for incident, polygon_names in sorted(current_global.items()):
            event_index = self._active_event_by_incident.get(incident)
            if event_index is None:
                _owner, obstacle_id = incident
                self._events.append(self._event_dict(
                    current_robot_ns_by_incident.get(incident, robot_ns),
                    self._EVENT_SOURCE,
                    polygon_names,
                    obstacle_id,
                ))
                self._active_event_by_incident[incident] = len(self._events) - 1
            else:
                self._events[event_index]["polygon_names"] = sorted(polygon_names)
                self._events[event_index]["polygon_name"] = sorted(polygon_names)[0] if polygon_names else ""

    def on_state(self, robot_ns: str, msg: CollisionMonitorState) -> None:
        if not self._active or robot_ns in self._robots_with_events:
            return
        current = {
            (msg.polygon_name, "<collision_monitor_state>")
        } if msg.polygon_name and msg.action_type == self._STOP_ACTION else set()
        self._record_current(robot_ns, self._STATE_SOURCE, current)

    def _record_current(
        self,
        robot_ns: str,
        source: str,
        current: set[tuple[str, str]],
    ) -> None:
        key = (robot_ns, source)
        previous = self._current_by_robot_source.get(key, set())
        for polygon_name, obstacle_id in sorted(current - previous):
            self._events.append(self._event_dict(robot_ns, source, {polygon_name}, obstacle_id))
        self._current_by_robot_source[key] = current

    def _drop_state_fallback(self, robot_ns: str) -> None:
        self._current_by_robot_source.pop((robot_ns, self._STATE_SOURCE), None)
        self._events = [
            event for event in self._events
            if not (
                event.get("robot_ns") == robot_ns
                and event.get("source") == self._STATE_SOURCE
            )
        ]

    @staticmethod
    def _event_dict(
        robot_ns: str,
        source: str,
        polygon_names: set[str],
        obstacle_id: str,
    ) -> dict[str, object]:
        robot_name = CollisionAccumulator._robot_name(robot_ns)
        sorted_polygon_names = sorted(polygon_names)
        return {
            "robot_name": robot_name,
            "robot_namespace": robot_ns,
            "robot_ns": robot_ns,
            "source": source,
            "polygon_name": sorted_polygon_names[0] if sorted_polygon_names else "",
            "polygon_names": sorted_polygon_names,
            "obstacle_id": obstacle_id,
        }

    def _global_event_contacts(self) -> dict[tuple[str, str], set[str]]:
        contacts: dict[tuple[str, str], set[str]] = {}
        for robot_contacts in self._event_contacts_by_robot.values():
            for incident, polygon_names in robot_contacts.items():
                contacts.setdefault(incident, set()).update(polygon_names)
        return contacts

    @staticmethod
    def _robot_name(robot_ns: str) -> str:
        return robot_ns.rstrip("/").rsplit("/", 1)[-1]

    @staticmethod
    def _collision_owner(robot_name: str, obstacle_id: str) -> str:
        if not obstacle_id.startswith("<robot:") or not obstacle_id.endswith(">"):
            return robot_name
        other = obstacle_id.removeprefix("<robot:").removesuffix(">")
        a, b = sorted((robot_name, other))
        return f"robot_pair:{a},{b}"

    @staticmethod
    def _canonical_obstacle_id(robot_name: str, obstacle_id: str) -> str:
        if not obstacle_id.startswith("<robot:") or not obstacle_id.endswith(">"):
            return obstacle_id
        other = obstacle_id.removeprefix("<robot:").removesuffix(">")
        a, b = sorted((robot_name, other))
        return f"<robot_pair:{a},{b}>"


class BenchmarkRunner(ArenaMixinNode):
    exit_code: typing.ClassVar[int] = 0

    def __init__(
        self,
        suite: Suite,
        contest: Contest,
        *,
        simulator: str | None,
        scale_episodes: float,
        env_n: int,
        run_id: str,
        headless: bool,
        run_dir: RunDir,
        retry_failed: bool = False,
        arena_passthrough: dict[str, str] | None = None,
        noexit: bool = False,
    ) -> None:
        super().__init__("arena_benchmark_runner")
        self._suite = suite
        self._contest = contest
        self._simulator = simulator
        self._scale_episodes = scale_episodes
        self._env_n = env_n
        self._run_id = run_id
        self._headless = headless
        self._run_dir = run_dir
        self._retry_failed = retry_failed
        self._arena_passthrough: dict[str, str] = dict(arena_passthrough or {})
        self._noexit = noexit
        self._total_groups = 0
        self._completed_groups = 0

        self._spawn = self.create_client_wrapper(SpawnEnv, "/arena/spawn_env")
        self._despawn = self.create_client_wrapper(DespawnEnv, "/arena/despawn_env")
        self._env_records: dict[int, EnvRecord] = {}
        self._env_gone_events: dict[int, asyncio.Event] = {}
        self._env_visible_events: dict[int, asyncio.Event] = {}

        self._episode_action_clients: dict[int, ActionClientWrapper] = {}
        self._queue_clients: dict[int, ClientWrapper] = {}
        self._episode_records: dict[int, dict[int, EpisodeRecord]] = {}
        self._collision_accumulators: dict[int, CollisionAccumulator] = {}
        self._collision_robot_topics: dict[int, set[str]] = {}
        self._env_ns_roots: dict[int, str] = {}
        self._env_subs: dict[int, list] = {}

        self.create_subscription(EnvRegistry, "/arena/state/envs", self._on_envs, _LATCHED)
        self._state_pub = self.create_publisher(BenchmarkState, STATE_TOPIC, _LATCHED)

        self._arena_proc: subprocess.Popen | None = None

    def _build_pending(self) -> list[Step]:
        return build_pending(
            suite=self._suite,
            contest=self._contest,
            scale_episodes=self._scale_episodes,
            run_dir=self._run_dir,
            retry_failed=self._retry_failed,
            record_root=self._run_dir.path,
        )

    def _on_envs(self, msg: EnvRegistry) -> None:
        new_ids = {e.env_id for e in msg.envs}
        for env_id in new_ids:
            self._env_visible_events.setdefault(env_id, asyncio.Event()).set()
        for env_id in list(self._env_gone_events):
            if env_id not in new_ids:
                self._env_gone_events[env_id].set()
        self._env_records = {e.env_id: e for e in msg.envs}

    def _build_launch_args(self, step: Step) -> list[str]:
        return build_launch_args(step, self._simulator)

    async def _await_env_visible(self, env_id: int) -> None:
        """Wait for env_id to appear on /arena/state/envs.

        No timeout. If the env never appears, this waits indefinitely (Ctrl+C to abort).
        """
        if env_id in self._env_records:
            return
        await self._env_visible_events.setdefault(env_id, asyncio.Event()).wait()

    async def _await_or_env_died(self, env_id: int, awaitable: typing.Awaitable[_T]) -> _T:
        """Race awaitable against env death. Raises _EnvDied if env_id disappears first.

        No-op (returns awaitable's result) if env_id is not currently registered (pre-spawn).
        """
        death = self._env_gone_events.setdefault(env_id, asyncio.Event())
        op_task = asyncio.ensure_future(awaitable)
        death_task = asyncio.ensure_future(death.wait())
        try:
            done, pending = await asyncio.wait(
                {op_task, death_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if death_task in done and op_task not in done:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise _EnvDied(f"env {env_id} disappeared from /arena/state/envs")
            if not death_task.done():
                death_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await death_task
            return op_task.result()
        except asyncio.CancelledError:
            op_task.cancel()
            death_task.cancel()
            raise

    async def _wait_env_gone(self, env_id: int, *, timeout: float | None) -> bool:
        ev = asyncio.Event()
        self._env_gone_events[env_id] = ev
        try:
            if env_id not in self._env_records:
                return True
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
        finally:
            self._env_gone_events.pop(env_id, None)

    async def _setup_env_clients(self, env_id: int, env_ns_root: str) -> None:
        """Create per-env action client, queue_episode client, and subscriptions. Idempotent."""
        if env_id in self._episode_action_clients:
            return

        action_name = f"{env_ns_root}/lifecycle/run_episode"
        self._episode_action_clients[env_id] = self.create_action_client_wrapper(
            RunEpisode, action_name
        )
        self._episode_records[env_id] = {}
        self._collision_accumulators[env_id] = CollisionAccumulator()
        self._collision_robot_topics[env_id] = set()
        self._env_ns_roots[env_id] = env_ns_root

        queue_client = self.create_client_wrapper(
            QueueEpisode, f"{env_ns_root}/config/queue_episode"
        )
        await queue_client.ensure(timeout_sec=30.0)
        self._queue_clients[env_id] = queue_client

        def _on_episode_record(msg: EpisodeRecord) -> None:
            recs = self._episode_records.get(env_id)
            if recs is not None:
                recs[msg.episode_id] = msg

        sub_ep = self.create_subscription(
            EpisodeRecord,
            f"{env_ns_root}/state/episode",
            _on_episode_record,
            10,
        )
        self._env_subs[env_id] = [sub_ep]

        def _on_robot_fleet(msg: RobotFleet) -> None:
            topics = self._collision_robot_topics.get(env_id)
            if topics is None:
                return
            for robot in msg.robots:
                self._ensure_collision_subscription(env_id, robot.ns.rstrip("/"))

        sub_robots = self.create_subscription(
            RobotFleet,
            f"{env_ns_root}/state/robots",
            _on_robot_fleet,
            _LATCHED,
        )
        self._env_subs[env_id].append(sub_robots)

    def _teardown_env_clients(self, env_id: int) -> None:
        """Destroy per-env subscriptions, action client, and queue_episode client."""
        for sub in self._env_subs.pop(env_id, []):
            self.destroy_subscription(sub)
        ac = self._episode_action_clients.pop(env_id, None)
        if ac is not None:
            ac.client.destroy()
        qc = self._queue_clients.pop(env_id, None)
        if qc is not None:
            qc.client.destroy()
        self._episode_records.pop(env_id, None)
        self._collision_accumulators.pop(env_id, None)
        self._collision_robot_topics.pop(env_id, None)
        self._env_ns_roots.pop(env_id, None)
        self._env_visible_events.pop(env_id, None)

    def _ensure_collision_subscription(self, env_id: int, robot_ns: str) -> None:
        topics = self._collision_robot_topics.get(env_id)
        if topics is None or not robot_ns:
            return
        topic = f"{robot_ns}/collision_events"
        if topic in topics:
            return
        topics.add(topic)

        def _on_collision_events(
            events_msg: CollisionEvents,
            *,
            _env_id: int = env_id,
            _robot_ns: str = robot_ns,
        ) -> None:
            acc = self._collision_accumulators.get(_env_id)
            if acc is not None:
                acc.on_events(_robot_ns, events_msg)

        self._env_subs.setdefault(env_id, []).append(
            self.create_subscription(CollisionEvents, topic, _on_collision_events, 10)
        )
        state_topic = f"{robot_ns}/collision_monitor_state"

        def _on_collision_state(
            state_msg: CollisionMonitorState,
            *,
            _env_id: int = env_id,
            _robot_ns: str = robot_ns,
        ) -> None:
            acc = self._collision_accumulators.get(_env_id)
            if acc is not None:
                acc.on_state(_robot_ns, state_msg)

        self._env_subs.setdefault(env_id, []).append(
            self.create_subscription(CollisionMonitorState, state_topic, _on_collision_state, 10)
        )

    def _ensure_stage_collision_subscriptions(self, env_id: int, robot_arg: str) -> None:
        env_ns_root = self._env_ns_roots.get(env_id)
        if not env_ns_root:
            return
        names = [name for name in str(robot_arg).split(",") if name]
        counts: dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        seen: dict[str, int] = {}
        for name in names:
            if counts[name] > 1:
                idx = seen.get(name, 0)
                seen[name] = idx + 1
                robot_name = f"{name}_{idx}"
            else:
                robot_name = name
            self._ensure_collision_subscription(env_id, f"{env_ns_root.rstrip('/')}/{robot_name}")

    async def _push_stage_config(self, env_id: int, step: Step) -> None:
        queue = self._queue_clients[env_id]
        req = QueueEpisode.Request()
        req.action = QueueEpisode.Request.MERGE
        req.world = step.stage.map
        req.tm_robots = step.stage.tm_robots.value
        req.tm_obstacles = step.stage.tm_obstacles.value
        req.tm_modules = []
        req.keep_modules = False
        req.robots = []
        obs_params, rob_params = _flatten_per_mode_params(
            step.stage.config,
            tm_obstacles=step.stage.tm_obstacles.value,
            tm_robots=step.stage.tm_robots.value,
        )
        req.obstacles_params = obs_params
        req.robots_params = rob_params
        resp = await self.await_ros(queue.client.call_async(req))
        if not resp.success:
            raise RuntimeError(f"queue_episode failed for {step.key}: {resp.error_msg}")

    async def _run_episodes(
        self,
        step: Step,
        env_id: int,
    ) -> StepResult:
        """Drive all episodes for one step. Env is already up and clients are set up."""
        started = time.time()
        episodes_run = 0
        episodes_failed = 0
        ac = self._episode_action_clients[env_id]

        try:
            self._ensure_stage_collision_subscriptions(env_id, step.stage.robot)
            for ep_idx in range(step.episodes):
                goal = RunEpisode.Goal()
                goal.world = step.stage.map
                goal.seed = step.stage.seed

                ep_started_sim = self.sim_time.to_seconds()
                ep_started_wall = time.time()
                collision_acc = self._collision_accumulators.get(env_id)
                if collision_acc is not None:
                    collision_acc.begin()

                goal_handle = await self._await_or_env_died(
                    env_id, ac.send_goal(goal)
                )

                try:
                    result_obj = await asyncio.wait_for(
                        self._await_or_env_died(env_id, ac.await_result(goal_handle)),
                        timeout=step.stage.timeout,
                    )
                except TimeoutError:
                    ep_ended_sim = self.sim_time.to_seconds()
                    collision_count, collision_events = (
                        collision_acc.end() if collision_acc is not None else (0, [])
                    )
                    episodes_failed += 1
                    self.get_logger().warning(
                        f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} "
                        f"TIMEOUT after {step.stage.timeout}s; "
                        f"collisions={collision_count}; cancelling and advancing"
                    )
                    with contextlib.suppress(Exception):
                        await self.await_ros(goal_handle.cancel_goal_async())
                    ts_iso = datetime.datetime.now(tz=datetime.UTC).isoformat()
                    timeout_record = make_timeout_episode_record(step, ep_idx)
                    self._run_dir.progress.append(
                        ts_iso=ts_iso,
                        run_id=self._run_id,
                        step_key=step.key,
                        contestant=step.contestant.name,
                        stage=step.stage.name,
                        env_id=env_id,
                        episode_id=ep_idx,
                        episode_record=timeout_record,
                        started_at=ep_started_sim,
                        ended_at=ep_ended_sim,
                        collision_count=collision_count,
                        collision_events=collision_events,
                        error_kind=StepErrorKind.EPISODE_TIMEOUT,
                        error_detail=f"stage.timeout exceeded ({step.stage.timeout}s)",
                    )
                    continue
                ep_ended_sim = self.sim_time.to_seconds()
                ep_ended_wall = time.time()
                collision_count, collision_events = (
                    collision_acc.end() if collision_acc is not None else (0, [])
                )

                result: RunEpisode.Result = result_obj.result
                episode_id = result.episode_id

                if result.state == RunEpisode.Result.FATAL:
                    self.get_logger().error(
                        f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} "
                        f"FATAL: {result.info} -- aborting step"
                    )
                    return StepResult(
                        step.key, "failed", env_id, started, time.time(),
                        StepErrorKind.ROBOT_SETUP, f"env reported FATAL: {result.info}",
                        episodes_run=episodes_run, episodes_failed=episodes_failed,
                        episodes_total=step.episodes,
                    )

                recs = self._episode_records.get(env_id, {})
                rec = recs.get(episode_id)
                if rec is None:
                    episodes_failed += 1
                    self.get_logger().warning(
                        f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} "
                        f"no EpisodeRecord for episode_id={episode_id}; counted as failed"
                    )
                    continue

                episodes_run += 1
                if rec.outcome_state == EpisodeRecord.FAILED:
                    episodes_failed += 1

                state_label = {
                    EpisodeRecord.SUCCESS: "SUCCESS",
                    EpisodeRecord.FAILED: "FAILED",
                    EpisodeRecord.SKIPPED: "SKIPPED",
                }.get(rec.outcome_state, str(rec.outcome_state))
                self.get_logger().info(
                    f"[{ep_idx + 1}/{step.episodes}] {step.key} env={env_id} "
                    f"{state_label} info={rec.outcome_info!r} "
                    f"collisions={collision_count} "
                    f"sim={ep_ended_sim - ep_started_sim:.1f}s "
                    f"wall={ep_ended_wall - ep_started_wall:.1f}s"
                )

                ts_iso = datetime.datetime.now(tz=datetime.UTC).isoformat()
                self._run_dir.progress.append(
                    ts_iso=ts_iso,
                    run_id=self._run_id,
                    step_key=step.key,
                    contestant=step.contestant.name,
                    stage=step.stage.name,
                    env_id=env_id,
                    episode_id=episode_id,
                    episode_record=rec,
                    started_at=ep_started_sim,
                    ended_at=ep_ended_sim,
                    collision_count=collision_count,
                    collision_events=collision_events,
                )
        except _EnvDied as exc:
            self.get_logger().warning(
                f"{step.key} env={env_id} env died mid-step after "
                f"run={episodes_run}, failed={episodes_failed}: {exc}"
            )
            return StepResult(
                step.key, "failed", env_id, started, time.time(),
                StepErrorKind.ENV_SETUP, repr(exc),
                episodes_run=episodes_run, episodes_failed=episodes_failed,
                episodes_total=step.episodes,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.get_logger().exception(
                f"{step.key} env={env_id} unexpected error mid-step after "
                f"run={episodes_run}, failed={episodes_failed}"
            )
            return StepResult(
                step.key, "failed", env_id, started, time.time(),
                StepErrorKind.INTERNAL, repr(exc),
                episodes_run=episodes_run, episodes_failed=episodes_failed,
                episodes_total=step.episodes,
            )

        if episodes_run == 0:
            status = "failed"
        elif episodes_failed == 0:
            status = "ok"
        elif episodes_failed < episodes_run:
            status = "partial"
        else:
            status = "failed"

        return StepResult(
            step.key, status, env_id, started, time.time(),
            None, None,
            episodes_run=episodes_run, episodes_failed=episodes_failed,
            episodes_total=step.episodes,
        )

    async def _run_group(self, group: list[Step], slot_index: int) -> list[StepResult]:
        env_id: int | None = None
        results: list[StepResult] = []
        try:
            req = SpawnEnv.Request()
            req.ns = ""
            req.headless = self._headless
            req.launch_args = self._build_launch_args(group[0])
            resp = await self.await_ros(self._spawn.client.call_async(req))
            if resp is None or not resp.success:
                msg = resp.error_msg if resp is not None else "no response"
                failed = StepResult(
                    group[0].key, "failed", None, time.time(), time.time(),
                    StepErrorKind.ENV_SETUP, f"spawn_env failed: {msg}",
                    episodes_total=group[0].episodes,
                )
                results.append(failed)
                for step in group[1:]:
                    results.append(StepResult(
                        step.key, "skipped", None, time.time(), time.time(),
                        StepErrorKind.CANCELLED,
                        "aborted by upstream step setup failure",
                        episodes_total=step.episodes,
                    ))
                return results
            env_id = resp.env_id

            await self._await_env_visible(env_id)
            env_ns_root = self._env_records[env_id].fqn

            await self._setup_env_clients(env_id, env_ns_root)

            for idx, step in enumerate(group):
                await self._push_stage_config(env_id, step)

                try:
                    step_result = await self._run_episodes(step, env_id)
                except asyncio.CancelledError:
                    step_result = StepResult(
                        step.key, "skipped", env_id, time.time(), time.time(),
                        StepErrorKind.CANCELLED, "cancelled",
                        episodes_total=step.episodes,
                    )
                    results.append(step_result)
                    for remaining in group[idx + 1:]:
                        results.append(StepResult(
                            remaining.key, "skipped", env_id, time.time(), time.time(),
                            StepErrorKind.CANCELLED, "cancelled",
                            episodes_total=remaining.episodes,
                        ))
                    raise

                results.append(step_result)

                if step_result.status == "failed" and step_result.error_kind in _SYSTEMIC:
                    for remaining in group[idx + 1:]:
                        results.append(StepResult(
                            remaining.key, "skipped", env_id, time.time(), time.time(),
                            StepErrorKind.CANCELLED,
                            "aborted by upstream step setup failure",
                            episodes_total=remaining.episodes,
                        ))
                    break

        except _EnvDied as exc:
            if not results:
                results.append(StepResult(
                    group[0].key, "failed", env_id, time.time(), time.time(),
                    StepErrorKind.ENV_SETUP, repr(exc),
                    episodes_total=group[0].episodes,
                ))
            already = {r.key for r in results}
            for step in group:
                if step.key not in already:
                    results.append(StepResult(
                        step.key, "skipped", env_id, time.time(), time.time(),
                        StepErrorKind.CANCELLED,
                        "aborted by upstream step setup failure",
                        episodes_total=step.episodes,
                    ))
        except asyncio.CancelledError:
            already = {r.key for r in results}
            for step in group:
                if step.key not in already:
                    results.append(StepResult(
                        step.key, "skipped", env_id, time.time(), time.time(),
                        StepErrorKind.CANCELLED, "cancelled",
                        episodes_total=step.episodes,
                    ))
            raise
        except Exception as exc:
            if not results:
                results.append(StepResult(
                    group[0].key, "failed", env_id, time.time(), time.time(),
                    StepErrorKind.INTERNAL, repr(exc),
                    episodes_total=group[0].episodes,
                ))
            already = {r.key for r in results}
            for step in group:
                if step.key not in already:
                    results.append(StepResult(
                        step.key, "skipped", env_id, time.time(), time.time(),
                        StepErrorKind.CANCELLED,
                        "aborted by upstream step setup failure",
                        episodes_total=step.episodes,
                    ))
        finally:
            self._completed_groups += 1
            keep_alive = (
                self._noexit
                and self._completed_groups == self._total_groups
                and env_id is not None
            )
            if env_id is not None:
                self._teardown_env_clients(env_id)
                if env_id in self._env_records and not keep_alive:
                    with contextlib.suppress(Exception):
                        dreq = DespawnEnv.Request()
                        dreq.env_id = env_id
                        await self._despawn.call_timeout(dreq, timeout_sec=30.0)
                    with contextlib.suppress(asyncio.TimeoutError, Exception):
                        await self._wait_env_gone(env_id, timeout=30.0)
                if keep_alive:
                    self.get_logger().info(
                        f"--noexit: keeping env {env_id} alive after last group {group[0].key}"
                    )
        return results

    def _publish_state(
        self,
        results: typing.Mapping[str, StepResult],
        steps_total: int,
    ) -> None:
        if not rclpy.ok():
            return
        msg = BenchmarkState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.run_id = self._run_id
        msg.suite = self._suite.name
        msg.contest = self._contest.name
        msg.simulator = self._simulator or ""
        msg.env_n = self._env_n
        msg.headless = self._headless
        msg.steps_total = steps_total
        msg.steps_done = sum(1 for r in results.values() if r.status == "ok")
        msg.steps_partial = sum(1 for r in results.values() if r.status == "partial")
        msg.steps_failed = sum(1 for r in results.values() if r.status == "failed")
        msg.steps_skipped = sum(1 for r in results.values() if r.status == "skipped")
        msg.steps_in_flight = sum(1 for r in results.values() if r.status == "in_progress")
        msg.active_keys = [k for k, r in results.items() if r.status == "in_progress"]
        self._state_pub.publish(msg)

    async def setup(self) -> None:
        try:
            BenchmarkRunner.exit_code = await self._run_steps()
        except Exception as exc:
            self.get_logger().error(f"benchmark crashed: {exc!r}")
            BenchmarkRunner.exit_code = 2
        finally:
            await self._shutdown_arena()
            rclpy.try_shutdown()

    async def teardown(self) -> None:
        await self._shutdown_arena()

    async def _shutdown_arena(self) -> None:
        if self._noexit:
            self.get_logger().info(
                "--noexit: leaving arena_runtime.launch.py running; Ctrl+C its terminal to stop"
            )
            return
        p = self._arena_proc
        if p is None or p.poll() is not None:
            return
        loop = asyncio.get_running_loop()
        for sig, grace in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0)):
            try:
                os.killpg(os.getpgid(p.pid), sig)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(loop.run_in_executor(None, p.wait), timeout=grace)
                return
            except TimeoutError:
                continue
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

    async def _run_steps(self) -> int:
        pending = self._build_pending()
        results: dict[str, StepResult] = dict(self._run_dir.state.steps)
        steps_total = len(results) + len(pending)
        aborted_systemic = False

        self._publish_state(results, steps_total)
        self.get_logger().info(f"benchmark: signalled READY on {STATE_TOPIC}")

        passthrough = dict(self._arena_passthrough)
        cmd = [
            "ros2", "launch", "arena_bringup", "arena_runtime.launch.py",
            *(f"{k}:={v}" for k, v in passthrough.items()),
        ]
        self._arena_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        await self._spawn.ensure(timeout_sec=300.0)
        await self._despawn.ensure(timeout_sec=300.0)

        groups = group_pending(pending, self._simulator)
        self._total_groups = len(groups)
        cap = max(1, min(self._env_n, len(groups) or 1))
        free_slots: list[int] = list(range(cap))
        in_flight: set[asyncio.Task[list[StepResult]]] = set()

        def _mark_group_in_progress(group: list[Step]) -> None:
            for step in group:
                results[step.key] = StepResult(
                    step.key, "in_progress", None, time.time(), None, None, None,
                    episodes_total=step.episodes,
                )
            self._run_dir.state.write(results)
            self._publish_state(results, steps_total)

        def _flush_group_results(group_results: list[StepResult]) -> bool:
            """Write results, return True only on a systemic setup failure before any episode ran."""
            for res in group_results:
                results[res.key] = res
                self.get_logger().info(
                    f"[{res.status}] {res.key} env={res.env_id} "
                    f"episodes={res.episodes_run}/{res.episodes_total} "
                    f"(failed={res.episodes_failed}) "
                    f"t={((res.ended_at or 0.0) - res.started_at):.1f}s"
                )
            self._run_dir.state.write(results)
            self._publish_state(results, steps_total)
            total_episodes_run = sum(r.episodes_run for r in results.values())
            if total_episodes_run > 0:
                return False
            return any(
                r.status == "failed" and r.error_kind in _SYSTEMIC for r in group_results
            )

        try:
            while groups or in_flight:
                while groups and len(in_flight) < cap and free_slots:
                    group = groups.pop(0)
                    slot = free_slots.pop(0)
                    _mark_group_in_progress(group)
                    task: asyncio.Task[list[StepResult]] = asyncio.create_task(
                        self._run_group(group, slot_index=slot),
                        name=group[0].key,
                    )
                    task.add_done_callback(lambda _t, s=slot: free_slots.append(s))
                    in_flight.add(task)
                if not in_flight:
                    break
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    group_results: list[StepResult] = t.result()
                    abort = _flush_group_results(group_results)
                    if abort:
                        aborted_systemic = True
                        first_failed = next(
                            r for r in group_results
                            if r.status == "failed" and r.error_kind in _SYSTEMIC
                        )
                        self.get_logger().error(
                            f"benchmark: {first_failed.key} hit a systemic setup failure "
                            f"({first_failed.error_kind}: {first_failed.error_detail}); "
                            f"aborting run before any episode ran, {len(groups)} pending group(s) skipped"
                        )
                        for t2 in in_flight:
                            t2.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(*in_flight, return_exceptions=True)
                        groups.clear()
                        in_flight.clear()
                        break
        except asyncio.CancelledError:
            for t in in_flight:
                t.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
            raise
        finally:
            self._run_dir.progress.dedupe_in_place()
            self._publish_state(results, steps_total)

        return 1 if aborted_systemic else 0


def _all_steps(contest: Contest, suite: Suite, scale_episodes: float) -> list[Step]:
    steps: list[Step] = []
    for contestant in contest.contestants:
        for stage in suite.stages:
            steps.append(
                Step(
                    contestant=contestant,
                    stage=stage,
                    episodes=int(round(stage.episodes * scale_episodes)),
                    record_dir=None,
                )
            )
    return steps


def _is_inline_contest(contest_name: str) -> bool:
    stripped = contest_name.strip()
    return stripped.startswith("[") or stripped.startswith("{")


def _is_inline_suite(suite_name: str) -> bool:
    stripped = suite_name.strip()
    return stripped.startswith("[") or stripped.startswith("{")


def _load_suite_contest(
    suite_name: str, contest_name: str
) -> tuple[Suite, Contest, dict, list | dict]:
    share = pathlib.Path(get_package_share_directory("arena_evaluation"))
    bench_dir = share / "configs" / "benchmark"
    if not bench_dir.exists():
        source_bench_dir = pathlib.Path(__file__).resolve().parents[2] / "configs" / "benchmark"
        if source_bench_dir.exists():
            bench_dir = source_bench_dir

    if _is_inline_suite(suite_name):
        suite_dict = yaml.safe_load(suite_name)
        suite = Suite.parse("inline", suite_dict)
    else:
        suite_stem = suite_name.removesuffix(".yaml")
        suite_path = bench_dir / "suites" / f"{suite_stem}.yaml"
        suite_dict = yaml.safe_load(suite_path.read_text())
        suite = Suite.parse(suite_stem, suite_dict)

    if _is_inline_contest(contest_name):
        contest_dict = yaml.safe_load(contest_name)
        contest = Contest.parse("inline", contest_dict)
    else:
        contest_stem = contest_name.removesuffix(".yaml")
        contest_path = bench_dir / "contests" / f"{contest_stem}.yaml"
        contest_dict = yaml.safe_load(contest_path.read_text())
        contest = Contest.parse(contest_stem, contest_dict)

    return suite, contest, suite_dict, contest_dict


def _default_run_id(suite_name: str, contest_name: str) -> str:
    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    if _is_inline_suite(suite_name):
        suite_stem = "inline"
    else:
        suite_stem = pathlib.Path(suite_name.removesuffix(".yaml")).stem
    if _is_inline_contest(contest_name):
        contest_stem = "inline"
    else:
        contest_stem = pathlib.Path(contest_name.removesuffix(".yaml")).stem
    return f"{ts}-{suite_stem}-{contest_stem}"


_KV_RE = re.compile(r"^\w+:=.*$")


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchmark")
    p.add_argument("--suite", default="basic")
    p.add_argument("--contest", default="basic")
    p.add_argument("--scale-episodes", type=float, default=1.0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument(
        "--resume",
        nargs="?",
        const="__auto__",
        default=None,
        help="Resume a prior run. Bare --resume picks the most recent resumable; "
             "--resume <run_id> opens that run explicitly.",
    )
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--noexit",
        action="store_true",
        help="On completion, leave arena_runtime.launch.py and the last env running so you "
             "can poke at it. Recording stops with the last episode as usual.",
    )
    args, extras = p.parse_known_args(argv)

    for arg in extras:
        if not _KV_RE.match(arg):
            p.error(f"unrecognized argument: {arg!r}")

    arena_passthrough: dict[str, str] = {}
    for arg in extras:
        k, v = arg.split(":=", 1)
        arena_passthrough[k] = v

    env_n = int(arena_passthrough.get("env_n", "1"))
    headless = arena_passthrough.get("headless", "false").lower() in ("true", "1")
    simulator = arena_passthrough.get("sim", None)

    try:
        suite, contest, suite_dict, contest_dict = _load_suite_contest(args.suite, args.contest)
        cfg_hash = compute_config_hash(suite_dict, contest_dict)

        share = pathlib.Path(get_package_share_directory("arena_evaluation"))

        if args.data_root:
            data_root = pathlib.Path(args.data_root)
            print(f"benchmark: data_root from --data-root: {data_root}", file=sys.stderr)
        elif os.environ.get("ARENA_DATA_DIR"):
            data_root = pathlib.Path(os.environ["ARENA_DATA_DIR"]) / "benchmarks"
            print(f"benchmark: data_root from ARENA_DATA_DIR: {data_root}", file=sys.stderr)
        else:
            data_root = share / "data"
            print(f"benchmark: data_root from default: {data_root}", file=sys.stderr)

        steps = _all_steps(contest, suite, args.scale_episodes)
        if not steps:
            print(
                f"benchmark: empty grid (suite={args.suite!r} contest={args.contest!r} "
                "produced no steps)",
                file=sys.stderr,
            )
            return 2
        seen: set[str] = set()
        for c in steps:
            if c.key in seen:
                print(f"benchmark: duplicate step key {c.key!r}", file=sys.stderr)
                return 2
            seen.add(c.key)

        if args.resume:
            resume_id = args.resume
            if resume_id == "__auto__":
                resolved = find_most_recent_resumable(data_root)
                if resolved is None:
                    print(
                        f"benchmark: no resumable runs in {data_root}",
                        file=sys.stderr,
                    )
                    return 2
                print(
                    f"benchmark: auto-resume picked run_id={resolved}",
                    file=sys.stderr,
                )
                resume_id = resolved
            run_dir = RunDir.open(data_root, resume_id)
            if run_dir.manifest.config_hash != cfg_hash and not args.force:
                print(
                    f"config_hash mismatch: run has {run_dir.manifest.config_hash!r}, "
                    f"current is {cfg_hash!r}. Pass --force to proceed anyway.",
                    file=sys.stderr,
                )
                return 2
            run_dir.progress.write_comment(
                f"resumed at {datetime.datetime.now(tz=datetime.UTC).isoformat()}"
            )
        else:
            run_id = args.run_id or _default_run_id(args.suite, args.contest)
            sha, dirty = capture_git_sha(share.parent.parent.parent)
            steps_list = [
                {
                    "key": c.key,
                    "contestant": attrs.asdict(c.contestant),
                    "stage": {
                        k: v.value if isinstance(v, (Constants.TaskMode.TM_Robots, Constants.TaskMode.TM_Obstacles)) else v
                        for k, v in c.stage._asdict().items()
                    },
                    "episodes_planned": c.episodes,
                }
                for c in steps
            ]
            manifest = Manifest(
                run_id=run_id,
                created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
                arena_git_sha=sha,
                arena_git_dirty=dirty,
                cli_args=sys.argv[1:] if argv is None else list(argv),
                env_n=env_n,
                headless=headless,
                config_hash=cfg_hash,
                simulator=simulator,
                scale_episodes=args.scale_episodes,
                suite_name=suite.name,
                contest_name=contest.name,
                suite=suite_dict,
                contest=contest_dict,
                steps=steps_list,
            )
            run_dir = RunDir.create(data_root, run_id, manifest)
    except FileNotFoundError as exc:
        print(f"benchmark: config file not found: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2

    print(
        f"benchmark: prepared run_id={run_dir.manifest.run_id} "
        f"steps={len(steps)} dir={run_dir.path}",
        file=sys.stderr,
    )

    try:
        BenchmarkRunner.run_main(
            suite=suite,
            contest=contest,
            simulator=simulator,
            scale_episodes=args.scale_episodes,
            env_n=env_n,
            run_id=run_dir.manifest.run_id,
            headless=headless,
            run_dir=run_dir,
            retry_failed=args.retry_failed,
            arena_passthrough=arena_passthrough,
            noexit=args.noexit,
        )
    except KeyboardInterrupt:
        return 130
    return BenchmarkRunner.exit_code


# Convenience: allow `python -m arena_evaluation.benchmark.runner`
if __name__ == "__main__":
    raise SystemExit(cli_main())
