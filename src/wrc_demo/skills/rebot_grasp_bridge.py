"""Bridge from wrc_demo skills to rebot_grasp motion + gripper APIs.

Maps wrc_demo's high-level skill calls (``skill_grasp_object``,
``skill_pick_and_place``) to rebot_grasp's ``RebotArmEndPose.move_to_traj``
and ``GraspDriver`` primitives so wrc_demo's trajectory matches
``rebot_grasp/scripts/main.py`` and ``set.py`` behavior.

Harness integration (pre/post ``harness.approve`` around ``move_to_traj``)
is layered on top in task B1; this slice is pure pass-through.

Reference (set.py lines 80-119 + 138-153):

    print("[Grasp] Open gripper")
    grasp_driver.open_gripper()

    print("[Grasp] Move to pregrasp")
    controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0)

    print("[Grasp] Move to grasp")
    controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5)

    print("[Grasp] Closing")
    ok = grasp_driver.grasp()

    print("[Grasp] Return ready")
    _move_ready(controller, ready_cfg)

For ``place``:

    grasp_driver.open_gripper(timeout=0.5)
    # background thread sleeps 0.8s then closes to idle (d=0)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np
    from reBotArm_control_py.controllers import RebotArmEndPose
    from rebot_grasp.drivers.robot.grasp_driver import GraspDriver
    from ..safety.harness import SafetyHarness


class RebotGraspBridge:
    """Thin facade exposing rebot_grasp API to wrc_demo skills.

    The bridge is constructed once at startup (in :func:`wrc_demo.control.make_arm`
    for the ``rebot_rs`` backend) and stashed on the wrapped ``SafeArm``. The
    skill runtime reads it as ``self.bridge``; when ``None`` (mock or non-RS
    backend) skills fall back to the legacy path.

    All methods are synchronous; ``release_with_async_close`` is the only one
    that schedules background work, and it returns the thread for the caller
    to ``.join()`` after the next ``move_to_traj``.

    Optional ``harness`` (:class:`wrc_demo.safety.harness.SafetyHarness`)
    integration adds:

    - ``move_to_traj`` is wrapped in ``begin_motion()`` / ``end_motion()``,
      so e-stop and the perception watchdog are checked at the boundary.
      Per-waypoint validation is delegated to the SDK controller (rebot_grasp
      doesn't expose joint-angle hooks; ``approve(q_prev, q_next)`` is only
      invoked from the wrc_demo SafeArm path).
    - Gripper methods call ``heartbeat()`` to refresh the watchdog.
    - ``SafetyViolation`` raised by ``begin_motion`` propagates; the underlying
      controller is not called.
    """

    def __init__(
        self,
        controller: "RebotArmEndPose",
        grasp_driver: "GraspDriver",
        harness: Optional["SafetyHarness"] = None,
    ) -> None:
        self._controller = controller
        self._grasp_driver = grasp_driver
        self._harness = harness
        self.motion_error: str | None = None

    # ──────────────────────────── motion ────────────────────────────

    def move_to_traj(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        duration: float,
    ) -> bool:
        """IK + min-jerk trajectory through ``RebotArmEndPose.move_to_traj``.

        Blocks until the motion thread completes (or times out), matching
        ``rebot_grasp/scripts/main.py``'s ``_wait_motion`` pattern.
        Without this wait, consecutive calls (pregrasp → grasp) would
        override each other — the arm never reaches pregrasp before being
        redirected to the grasp target, causing table collisions and
        empty grasps.

        Returns ``True`` once the sender completes. The caller verifies
        the physical endpoint separately using joint feedback.

        With ``harness`` set, calls ``begin_motion()`` first (e-stop +
        watchdog check) and ``end_motion()`` after; ``SafetyViolation`` from
        ``begin_motion`` propagates without invoking the controller.
        """
        self.motion_error = None
        if self._harness is not None:
            self._harness.begin_motion()
        try:
            if not self._controller.move_to_traj(
                x, y, z, roll, pitch, yaw, duration=duration,
            ):
                self.motion_error = self._sdk_motion_error() or "SDK rejected trajectory"
                return False

            timeout = duration + 2.6
            if not self._wait_motion(duration, timeout):
                self.motion_error = self._sdk_motion_error()
                if self.motion_error is None:
                    self.motion_error = f"SDK send thread timed out after {timeout:.2f} s"
                    cancel = getattr(self._controller, "cancel_trajectory", None)
                    if callable(cancel):
                        cancel()
                return False
            return True
        finally:
            if self._harness is not None:
                self._harness.end_motion()

    def get_commanded_joints(self) -> np.ndarray | None:
        read_target = getattr(self._controller, "get_commanded_joints", None)
        return read_target() if callable(read_target) else None

    def _sdk_motion_error(self) -> str | None:
        reason = getattr(self._controller, "motion_error", None)
        return reason if isinstance(reason, str) else None

    def _wait_motion(self, duration: float, timeout: float) -> bool:
        """Block until the SDK motion thread finishes."""
        wait_fn = getattr(self._controller, "wait_motion_done", None)
        if callable(wait_fn):
            return bool(wait_fn(timeout=timeout))

        thread = getattr(self._controller, "_send_thread", None)
        if thread is not None:
            thread.join(timeout=timeout)
            return not thread.is_alive()

        import time
        time.sleep(duration + 0.6)
        return True

    # ──────────────────────────── gripper ───────────────────────────

    def open_gripper(self, timeout: Optional[float] = None) -> None:
        """Open to ``GraspDriver.MAX_DISTANCE_M``.

        ``timeout`` is forwarded only when supplied, so :class:`GraspDriver`'s
        own default (``3.0 s``) is used otherwise. ``set.py`` calls with
        ``timeout=0.5`` on place release; pregrasp uses the default.

        With ``harness`` set, refreshes the perception watchdog first.
        """
        if self._harness is not None:
            self._harness.heartbeat()
        if timeout is not None:
            self._grasp_driver.open_gripper(timeout=timeout)
        else:
            self._grasp_driver.open_gripper()

    def grasp(self, force: Optional[float] = None) -> bool:
        """Single-stage close with force control.

        Stops on object contact (``|vel| < stall_vel``) or hard stop; returns
        ``True`` if a held object was detected. ``force`` is the hold-torque
        magnitude; ``set.py`` uses default ``0.30``, we override with
        ``1.0`` to match wrc_demo v5's ``effort=1.00``.

        With ``harness`` set, refreshes the perception watchdog first.
        """
        if self._harness is not None:
            self._harness.heartbeat()
        if force is not None:
            return bool(self._grasp_driver.grasp(force=force))
        return bool(self._grasp_driver.grasp())

    def release_gripper(self) -> None:
        """Open + return to closed rest synchronously (used at demo exit).

        With ``harness`` set, refreshes the perception watchdog first.
        """
        if self._harness is not None:
            self._harness.heartbeat()
        self._grasp_driver.release_gripper()

    def release_with_async_close(
        self,
        timeout: float = 0.5,
        delay_s: float = 0.8,
    ) -> threading.Thread:
        """Open immediately, close to idle ``delay_s`` later in background.

        Mirrors ``set.py`` lines 138-149: the falling banana leaves the
        fingers before the close starts, so it doesn't snag. The caller
        must ``.join()`` the returned thread after the next ``move_to_traj``
        so the close completes before the next skill's grasp.

        Note: as of ADR-0005 ``set_position_target`` is a **public**
        method on ``WrcGripper`` (was ``_set_position_target`` on the
        upstream ``GraspDriver``). The try/except remains so a future
        rename degrades silently — but it's defensive only, since we
        own both sides of the API now.
        """
        self._grasp_driver.open_gripper(timeout=timeout)

        driver = self._grasp_driver

        def _close_after_delay():
            import time
            time.sleep(delay_s)
            try:
                driver.set_position_target(0.0)
            except AttributeError:
                # Defensive: rename-safe. The gripper stays at the
                # last commanded position (post-open); the next grasp
                # or open call will reset the state machine.
                pass

        thread = threading.Thread(target=_close_after_delay, daemon=True)
        thread.start()
        return thread
