from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time


def _data_root() -> pathlib.Path:
    env = os.environ.get("ARENA_DATA_DIR")
    if env:
        return pathlib.Path(env) / "benchmarks"
    from ament_index_python.packages import get_package_share_directory

    return pathlib.Path(get_package_share_directory("arena_evaluation")) / "data"


def _resolve_run(data_root: pathlib.Path, run_id: str | None) -> pathlib.Path:
    if run_id:
        path = data_root / run_id
        if not path.is_dir():
            raise SystemExit(f"no run found at {path}")
        return path
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist: {data_root}")
    runs = sorted([p for p in data_root.iterdir() if p.is_dir()], reverse=True)
    if not runs:
        raise SystemExit(f"no benchmark runs in {data_root}")
    return runs[0]


def _count_by_status(steps: dict) -> dict[str, int]:
    counts: dict[str, int] = {
        "ok": 0,
        "partial": 0,
        "failed": 0,
        "skipped": 0,
        "in_progress": 0,
    }
    for step in steps.values():
        s = step.status
        if s in counts:
            counts[s] += 1
    return counts


def _cmd_list(args: argparse.Namespace) -> int:
    from .state import Manifest, StateFile

    data_root = pathlib.Path(args.data_root) if args.data_root else _data_root()

    if not data_root.is_dir():
        print(f"no benchmark runs in {data_root}")
        return 0

    runs = sorted([p for p in data_root.iterdir() if p.is_dir()], reverse=True)
    if not runs:
        print(f"no benchmark runs in {data_root}")
        return 0

    rows: list[tuple[str, str, str, int, int, int, int, int, int, str]] = []
    for run_path in runs:
        manifest_path = run_path / "manifest.yaml"
        if not manifest_path.exists():
            continue
        try:
            manifest = Manifest.from_yaml(manifest_path.read_text())
        except Exception:
            continue
        state = StateFile.open(run_path)
        counts = _count_by_status(state.steps)
        total = len(manifest.steps)
        created = manifest.created_at[:16].replace("T", " ") if manifest.created_at else ""
        rows.append((
            manifest.run_id,
            manifest.suite_name,
            manifest.contest_name,
            total,
            counts["ok"],
            counts["partial"],
            counts["failed"],
            counts["skipped"],
            counts["in_progress"],
            created,
        ))

    if not rows:
        print(f"no benchmark runs in {data_root}")
        return 0

    col_widths = [
        max(len("RUN_ID"), max(len(r[0]) for r in rows)),
        max(len("SUITE"), max(len(r[1]) for r in rows)),
        max(len("CONTEST"), max(len(r[2]) for r in rows)),
        len("STEPS"),
        len("OK"),
        len("PARTIAL"),
        len("FAILED"),
        len("SKIPPED"),
        len("IN_FLIGHT"),
        len("CREATED"),
    ]

    def _row(
        run_id: str,
        suite: str,
        contest: str,
        total: int | str,
        ok: int | str,
        partial: int | str,
        failed: int | str,
        skipped: int | str,
        in_flight: int | str,
        created: str,
    ) -> str:
        return (
            f"{str(run_id):<{col_widths[0]}}  "
            f"{str(suite):<{col_widths[1]}}  "
            f"{str(contest):<{col_widths[2]}}  "
            f"{str(total):>{col_widths[3]}}  "
            f"{str(ok):>{col_widths[4]}}  "
            f"{str(partial):>{col_widths[5]}}  "
            f"{str(failed):>{col_widths[6]}}  "
            f"{str(skipped):>{col_widths[7]}}  "
            f"{str(in_flight):>{col_widths[8]}}  "
            f"{str(created)}"
        )

    print(_row("RUN_ID", "SUITE", "CONTEST", "STEPS", "OK", "PARTIAL", "FAILED", "SKIPPED", "IN_FLIGHT", "CREATED"))
    for r in rows:
        print(_row(*r))

    return 0


def _format_status_block(
    run_id: str,
    suite: str,
    contest: str,
    simulator: str,
    env_n: int,
    headless: bool,
    created_at: str,
    steps_total: int,
    ok: int,
    partial: int,
    failed: int,
    skipped: int,
    in_flight: int,
    active: list[tuple[str, str | None]],
    failed_steps: list[tuple[str, str | None, str | None]],
) -> str:
    pending = steps_total - ok - partial - failed - skipped - in_flight
    lines = [
        f"run: {run_id}",
        f"suite/contest: {suite}/{contest}",
        f"simulator: {simulator}    env_n: {env_n}    headless: {headless}",
        f"created: {created_at}",
        "",
        f"steps: {steps_total}    ok: {ok}  partial: {partial}  failed: {failed}  skipped: {skipped}  in_flight: {in_flight}  pending: {pending}",
    ]
    if active:
        lines.append("")
        lines.append("active:")
        for key, started in active:
            if started is not None:
                lines.append(f"  {key} (started {started})")
            else:
                lines.append(f"  {key}")
    if failed_steps:
        lines.append("")
        lines.append("failed:")
        for key, kind, detail in failed_steps:
            lines.append(f"  {key}: {kind or 'unknown'}: {detail or ''}")
    return "\n".join(lines)


def _ago(ts: float | None) -> str | None:
    if ts is None:
        return None
    delta = int(time.time() - ts)
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    return f"{delta // 3600}h ago"


def _status_from_disk(data_root: pathlib.Path, run_id: str | None) -> str:
    from .state import Manifest, StateFile

    run_path = _resolve_run(data_root, run_id)
    manifest_path = run_path / "manifest.yaml"
    manifest = Manifest.from_yaml(manifest_path.read_text())
    state = StateFile.open(run_path)
    counts = _count_by_status(state.steps)

    active: list[tuple[str, str | None]] = []
    failed_steps: list[tuple[str, str | None, str | None]] = []
    for step in state.steps.values():
        if step.status == "in_progress":
            active.append((step.key, _ago(step.started_at)))
        elif step.status == "failed":
            kind = step.error_kind.value if step.error_kind is not None else None
            failed_steps.append((step.key, kind, step.error_detail))

    return _format_status_block(
        run_id=manifest.run_id,
        suite=manifest.suite_name,
        contest=manifest.contest_name,
        simulator=manifest.simulator or "",
        env_n=manifest.env_n,
        headless=manifest.headless,
        created_at=manifest.created_at,
        steps_total=len(manifest.steps),
        ok=counts["ok"],
        partial=counts["partial"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        in_flight=counts["in_progress"],
        active=active,
        failed_steps=failed_steps,
    )


def _cmd_status(args: argparse.Namespace) -> int:
    data_root = pathlib.Path(args.data_root) if args.data_root else _data_root()
    run_id: str | None = args.run_id

    if not args.watch:
        print(_status_from_disk(data_root, run_id))
        return 0

    import rclpy
    from arena_evaluation_msgs.msg import BenchmarkState
    from arena_rclpy_mixins import ArenaMixinNode, run_main

    class _WatchNode(ArenaMixinNode):
        def __init__(self) -> None:
            super().__init__("evaluation_cli_watch")
            self.create_subscription(
                BenchmarkState,
                "/arena/benchmark/state",
                self._on_state,
                rclpy.qos.QoSProfile(
                    depth=1,
                    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                ),
            )

        def _on_state(self, msg: BenchmarkState) -> None:
            active = [(k, None) for k in msg.active_keys]
            block = _format_status_block(
                run_id=msg.run_id,
                suite=msg.suite,
                contest=msg.contest,
                simulator=msg.simulator,
                env_n=msg.env_n,
                headless=msg.headless,
                created_at="",
                steps_total=msg.steps_total,
                ok=msg.steps_done,
                partial=msg.steps_partial,
                failed=msg.steps_failed,
                skipped=msg.steps_skipped,
                in_flight=msg.steps_in_flight,
                active=active,
                failed_steps=[],
            )
            print("\033[2J\033[H", end="")
            print(block)

    try:
        run_main(_WatchNode)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_tail(args: argparse.Namespace) -> int:
    data_root = pathlib.Path(args.data_root) if args.data_root else _data_root()
    run_path = _resolve_run(data_root, args.run_id)
    csv_path = run_path / "progress.csv"

    while not csv_path.exists():
        print(f"progress.csv not yet created at {csv_path}; waiting...")
        time.sleep(1)

    try:
        subprocess.run(["tail", "-n", "50", "-F", str(csv_path)], check=False)
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation_cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list benchmark runs")
    p_list.add_argument("--data-root", default=None, metavar="PATH")

    p_status = sub.add_parser("status", help="show run status")
    p_status.add_argument("--data-root", default=None, metavar="PATH")
    p_status.add_argument("--watch", action="store_true", help="subscribe to live topic")
    p_status.add_argument("run_id", nargs="?", default=None)

    p_tail = sub.add_parser("tail", help="tail progress.csv of a run")
    p_tail.add_argument("--data-root", default=None, metavar="PATH")
    p_tail.add_argument("run_id", nargs="?", default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "tail":
            return _cmd_tail(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
