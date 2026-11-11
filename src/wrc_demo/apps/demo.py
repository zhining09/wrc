"""Main demo entry point.

    wrc-demo --task "pick and place pink object" \
             --cameras l515,uvc --arm rebot_rs --llm anthropic

Defaults are the fully-offline stack (mock camera/arm/llm) so the wiring can
always be exercised without hardware or network. `--interactive` keeps the
session open for multiple tasks with persistent memory and beliefs.

Every run now livestreams: N cameras pump continuously (CameraRig), the
WorldWatcher keeps the belief store warm, and an MJPEG dashboard serves the
whole rig to any browser. Routine commands run on the reflex/experience fast
path without an LLM call; only novel tasks reach the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..agent.advisor import Advisor
from ..agent.llm import make_llm
from ..agent.orchestrator import AgentOrchestrator
from ..agent.reflex import ExperienceMemory, FastPlanner
from ..agent.trace import TraceLogger
from ..config import Cfg, PACKAGE_ROOT, load_demo_config
from ..config import resolve_llm_profile as _resolve_llm_profile
from ..control import make_arm
from ..memory import BeliefStore, EpisodicMemory
from ..perception.camera_base import make_camera
from ..perception.depth_provider import DepthProvider
from ..perception.grounding import Extrinsics
from ..perception.stream import CameraRig, CameraStream
from ..perception.world import LockedDetector, WatchedCamera, WorldWatcher
from ..safety.harness import Kinematics, SafetyHarness, SafetyLimits
from ..skills.runtime import SkillRuntime


def _camera_cfgs(cfg) -> list[Cfg]:
    """All camera profiles (cfg.cameras when present, else [cfg.camera])."""
    raw = cfg.get("cameras")
    if raw:
        return [Cfg(c) if isinstance(c, dict) else c for c in raw]
    return [cfg.camera]


def build_runtime(
    cfg,
    run_dir: Path,
    view: bool = False,
    lazy_arm: bool = False,
    serve: bool = False,
) -> tuple[SkillRuntime, object]:
    kin = Kinematics(
        urdf_path=str(cfg.arm.urdf),
        ee_frame=str(cfg.arm.get("ee_frame", "gripper_end")),
        arm_joint_count=int(cfg.arm.get("n_joints", 6)),
    )
    # ── the camera rig: N continuous streams, first = manipulation ──────
    cam_cfgs = _camera_cfgs(cfg)
    streams, watched = [], []
    # detector init moved AFTER rig.open() to avoid USB reset during YOLO load

    def fk():
        # Eye-in-hand extrinsics need live FK -- but the 3 Hz watcher must
        # NEVER be the thing that triggers an arm re-connect (the SDK's
        # start() powers the motors). The watcher skips this camera until
        # the arm reports connected.
        if not getattr(arm, "connected", True):
            from ..types import SkillError

            raise SkillError("eye-in-hand extrinsics need the arm (still in standby)")
        return kin.fk(arm.get_state().q)

    for i, ccfg in enumerate(cam_cfgs):
        extr_cfg = ccfg.get("extrinsics", _empty_cfg())
        fuse = bool(ccfg.get("fuse_beliefs", "extrinsics" in ccfg))
        if ccfg.get("requires_calibration", False):
            source = extr_cfg.get("source")
            legacy_source = extr_cfg.get("hand_eye_npz")
            path = next((p for p in (source, legacy_source) if p and Path(p).is_file()), None)
            real_primary = i == 0 and cfg.arm.type != "mock"
            if not path:
                message = f"Camera {ccfg.name!r} needs a hand-eye calibration at {source or legacy_source!r}"
                if real_primary:
                    raise ValueError(f"{message} before running with a real arm. Run wrc-calib-top first.")
                print(f"[wrc-demo] {message}; fusing beliefs is disabled", file=sys.stderr)
                fuse = False
            elif real_primary and path == source:
                serial = json.loads(Path(path).read_text()).get("camera_serial")
                if serial and serial != "REDACTED":
                    configured_serial = ccfg.get("serial")
                    if configured_serial and str(configured_serial) != str(serial):
                        raise ValueError(
                            f"Camera {ccfg.name!r} is configured for serial {configured_serial!r}, "
                            f"but {path} was calibrated for {serial!r}. Run wrc-calib-top again."
                        )
                    ccfg = Cfg({**ccfg.as_dict(), "serial": str(serial)})
        stream = CameraStream(
            make_camera(ccfg),
            name=str(ccfg.get("name", f"cam{i}")),
            rate_hz=float(ccfg.get("fps", 30.0)),
        )
        streams.append(stream)
        watched.append(
            WatchedCamera(
                stream=stream,
                depth=DepthProvider(ccfg),
                extrinsics=Extrinsics.from_config(extr_cfg, fk_tcp2base=fk),
                # A camera without calibrated extrinsics must not fuse 3D
                # beliefs (garbage base-frame positions); it still streams
                # video + overlays + heartbeats.
                fuse=fuse,
            )
        )
    rig = CameraRig(streams)
    rig.open()
    try:
        # Detector init AFTER camera open to avoid USB reset during YOLO/GPU load.
        detector = LockedDetector(_make_detector(cfg))
        rig.primary.warm_up(int(cam_cfgs[0].get("warmup_frames", 5)))
        # make_arm powers on a real arm after the camera is ready.
        arm = make_arm(cfg, kinematics=kin)
    except Exception:
        rig.close()  # a partial build must not leak open camera streams
        raise

    memory = EpisodicMemory(horizon_s=float(cfg.memory.get("horizon_s", 15.0)))
    beliefs = BeliefStore()
    trace = TraceLogger(run_dir)
    runtime = SkillRuntime(
        rig.primary, watched[0].depth, detector, watched[0].extrinsics,
        kin, arm, memory, beliefs, trace, cfg,
    )
    runtime.rig = rig

    pcfg = cfg.get("perception_loop", _empty_cfg())
    if bool(pcfg.get("enabled", True)):
        watcher = WorldWatcher(
            watched, detector, beliefs,
            classes=list(cfg.get("detect_classes", [])) or ["cup", "bottle", "box"],
            rate_hz=float(pcfg.get("rate_hz", 3.0)),
            harness=arm.harness,  # MockArm / SafeArm both expose .harness
        )
        watcher.start()
        runtime.watcher = watcher

    runtime.stream_server = None
    scfg = cfg.get("stream", _empty_cfg())
    if serve and bool(scfg.get("enabled", True)):
        from .stream_server import StreamServer

        server = StreamServer(
            rig,
            state_fn=lambda: _runtime_state(runtime),
            host=str(scfg.get("host", "0.0.0.0")),
            port=int(scfg.get("port", 8090)),
            fps=float(scfg.get("fps", 15.0)),
            quality=int(scfg.get("quality", 80)),
        )
        try:
            server.start()
            runtime.stream_server = server
        except OSError as e:  # port taken: the demo must keep running
            print(f"[wrc-demo] livestream disabled ({e}); "
                  "set stream.port or WRC_STREAM_PORT", file=sys.stderr)

    if view:
        from .live_view import RigViewer

        runtime.viewer = RigViewer(rig)
        runtime.viewer.start()

    return runtime, arm


def _runtime_state(runtime) -> dict:
    """Live world state for the dashboard/MCP (never touches the lazy arm)."""
    _, status = runtime.camera.overlay() if hasattr(runtime.camera, "overlay") else ([], "n/a")
    out = {
        "agent_status": status,
        "task": runtime.current_task,
        "holding": runtime.held_object,
        "objects": runtime.beliefs.summary(),
        "arm_connected": getattr(runtime.arm, "connected", True),
        # human-readable narration: newest events last (observations, skill
        # calls, outcomes) -- the dashboard renders this as the activity feed
        "events": runtime.memory.digest(max_lines=14).splitlines(),
    }
    if runtime.watcher is not None:
        out["perception"] = runtime.watcher.stats()
    return out


def shutdown_runtime(runtime, arm) -> None:
    """Stop threads and hardware in dependency order; never raises."""
    import contextlib

    for step in (
        lambda: runtime.watcher.stop() if runtime.watcher is not None else None,
        lambda: runtime.stream_server.stop() if getattr(runtime, "stream_server", None) else None,
        lambda: runtime.viewer.stop() if getattr(runtime, "viewer", None) else None,
        lambda: runtime.rig.close() if getattr(runtime, "rig", None) else runtime.camera.close(),
        arm.disconnect,
    ):
        with contextlib.suppress(Exception, KeyboardInterrupt):
            step()


def _empty_cfg():
    from ..config import Cfg

    return Cfg({})


def _make_detector(cfg):
    # A camera profile may pin its own detector (the mock camera uses the
    # mock detector so offline runs never load model weights).
    dcfg = cfg.camera.get("detector") or cfg.detector
    if dcfg.type == "mock":
        from ..perception.detector import MockDetector

        return MockDetector(label=dcfg.get("label", "red cube"))
    from ..perception.detector import OpenVocabDetector

    return OpenVocabDetector(
        model_path=dcfg.model,
        device=dcfg.get("device", "cuda:0"),
        conf=float(dcfg.get("conf", 0.25)),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WRC agentic grasping demo")
    p.add_argument("--task", default=None, help="natural-language task")
    p.add_argument("--camera", help="camera profile (overrides demo.yaml camera.profile for a real arm)")
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera profiles; first = manipulation "
                        "camera (e.g. l515,uvc). Overrides --camera.")
    p.add_argument("--arm", default="mock", help="arm profile (mock|rebot_rs)")
    p.add_argument("--llm", default=None,
               help="llm profile (overrides demo.yaml llm.profile); "
                    "filename stem from configs/llm/ (e.g. kimi_code or local_qwen3_vl)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--run-dir", default=None, help="trace output dir")
    p.add_argument("--interactive", action="store_true", help="multi-task REPL")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--no-serve", action="store_true",
                   help="do not start the MJPEG livestream dashboard")
    args = p.parse_args(argv)

    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    llm_profile = _resolve_llm_profile(args.llm)
    cfg = load_demo_config(camera=args.camera, cameras=cameras, arm=args.arm, llm=llm_profile)
    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    )
    import os

    view = not args.no_view and bool(os.environ.get("DISPLAY"))
    print(f"[wrc-demo] cameras={[cam.name for cam in _camera_cfgs(cfg)]} arm={args.arm} "
          f"llm={llm_profile} view={view}")
    print(f"[wrc-demo] traces -> {run_dir}")

    llm = make_llm(cfg.llm)
    runtime, arm = build_runtime(cfg, run_dir, view=view, serve=not args.no_serve)
    if runtime.stream_server is not None:
        print(f"[wrc-demo] LIVESTREAM dashboard: {runtime.stream_server.local_url}")

    # Ctrl+C = soft stop (freeze + latch e-stop, no free-fall); a second
    # Ctrl+C raises KeyboardInterrupt and tears the process down.
    import signal

    def _sigint(_sig, _frm):
        print("\n[wrc-demo] SIGINT: soft-stopping the arm (Ctrl+C again to exit)")
        runtime.arm.stop()
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _sigint)
    from ..agent.llm import MockLLM

    is_mock = isinstance(llm, MockLLM)
    advisor = Advisor(llm) if (llm.supports_vision and not is_mock) else None
    experience = ExperienceMemory(PACKAGE_ROOT / "runs" / "experience.json")
    # ADR-0008: optional LLM fallback chain. Read from cfg._data
    # (sanctioned per CLAUDE.md §Config system) so a YAML block like
    #   llm:
    #     fallback: [local_qwen3_vl, mock]
    # wires without any new API surface. Default empty chain = the
    # pre-ADR-0008 behavior (raise on primary failure).
    fallback_chain = list(cfg._data.get("llm", {}).get("fallback", []) or [])
    agent = AgentOrchestrator(
        llm, runtime, advisor=advisor, max_steps=args.max_steps,
        decompose=not is_mock, fast_planner=FastPlanner(experience),
        cfg=cfg, fallback_chain=fallback_chain,
    )

    def _run(task: str):
        runtime.current_task = task
        try:
            return agent.run_task(task)
        finally:
            runtime.current_task = None

    # Wire the dashboard "send"/"stop" buttons to the live agent so you can
    # drive the robot from the browser (http://<ip>:8090) as well as the REPL.
    _srv = getattr(runtime, "stream_server", None)
    if _srv is not None:
        _srv.set_task_fn(lambda t: _print_report(_run(t)))
        _srv.set_cancel_fn(runtime.arm.stop)

    try:
        if args.interactive:
            print("Type a task (empty line to quit).")
            while True:
                try:
                    task = input("task> ").strip()
                except EOFError:
                    break
                if not task:
                    break
                # Clear e-stop before each task so a previous Ctrl+C
                # does not permanently block motion.
                if hasattr(runtime, "arm") and hasattr(runtime.arm, "harness"):
                    runtime.arm.harness.reset_estop()
                _print_report(_run(task))
        else:
            task = args.task or "look at the table and report what objects you see"
            report = _run(task)
            _print_report(report)
            return 0 if report.success else 1
    finally:
        shutdown_runtime(runtime, arm)
    return 0


def _print_report(report) -> None:
    print("\n=== task report ===")
    print(f"task:    {report.task}")
    print(f"success: {report.success}")
    print(f"path:    {report.path} ({report.duration_s}s)")
    print(f"steps:   {report.steps}")
    print(f"summary: {report.summary}")
    # 2026-08-13: explicit "task done" signal — without this users see the
    # arm still hovering at ready pose and (incorrectly) assume the task
    # is still running, then re-issue the same command and trigger a fresh
    # pick_and_place cycle on the object they just placed.
    status = "✓ done" if report.success else "✗ failed"
    print(f"\n[{status}] task complete — arm parked at home. "
          f"Enter a new task or press Enter to quit.")


if __name__ == "__main__":
    sys.exit(main())
