from __future__ import annotations

import enum
import pathlib
import typing

import attrs

from .config import Contest, Suite


@attrs.frozen
class Step:
    contestant: Contest.Contestant
    stage: Suite.Stage
    episodes: int
    record_dir: pathlib.Path | None

    @property
    def key(self) -> str:
        return f"{self.contestant.name}/{self.stage.name}"


class StepErrorKind(enum.StrEnum):
    ENV_SETUP = "env_setup"
    ROBOT_SETUP = "robot_setup"
    EPISODE_TIMEOUT = "episode_timeout"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@attrs.frozen
class StepResult:
    key: str
    status: typing.Literal["ok", "failed", "skipped", "partial", "in_progress"]
    env_id: int | None
    started_at: float
    ended_at: float | None
    error_kind: StepErrorKind | None
    error_detail: str | None
    episodes_run: int = 0
    episodes_failed: int = 0
    episodes_total: int = 0
