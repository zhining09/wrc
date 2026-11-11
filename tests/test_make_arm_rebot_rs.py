"""Tests for make_arm rebot_rs wiring the RebotGraspBridge.

The rebot_rs backend must (a) instantiate ``WrcGripper`` (wrc_demo's
native port of rebot_grasp's GraspDriver — see ADR-0005) so the bridge
has a real gripper handle, (b) start the SDK control loop once
(WrcGripper.start() cooperates with the loop via ``_loop_cb``
wrapper; calling ``sdk_ctrl.start()`` afterwards would raise
RuntimeError on the second ``start_control_loop``), (c) attach a
``RebotGraspBridge`` to the returned ``SafeArm`` so skill runtime can
read ``arm.grasp_bridge``.

These tests monkeypatch the heavy SDK imports so they run without
real hardware.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wrc_demo.config import load_demo_config


def test_rebot_rs_arm_has_grasp_bridge_attribute():
    """make_arm for rebot_rs returns an arm with non-None ``grasp_bridge``."""
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")

    fake_arm = MagicMock()
    fake_ctrl = MagicMock()
    fake_ctrl._running = False
    fake_gripper = MagicMock()
    fake_gripper.MAX_DISTANCE_M = 0.09

    with patch("reBotArm_control_py.actuator.RebotArm", return_value=fake_arm), \
         patch(
             "reBotArm_control_py.controllers.RebotArmEndPose",
             return_value=fake_ctrl,
         ), \
         patch(
             "wrc_demo.control.gripper.WrcGripper",
             return_value=fake_gripper,
         ):
        # Lazy import so the patches apply before module import.
        from wrc_demo.control import make_arm

        arm = make_arm(cfg, kinematics=MagicMock())

    assert hasattr(arm, "grasp_bridge"), (
        "SafeArm must have a grasp_bridge attribute set by make_arm"
    )
    assert arm.grasp_bridge is not None


def test_rebot_rs_wrc_gripper_start_called_not_ctrl_start():
    """WrcGripper.start() cooperates with the SDK loop via _loop_cb wrapper.

    The control loop is started exactly once: by the rebot_rs path,
    sdk_ctrl.start() is NOT called (WrcGripper extends the loop, not
    restarts it).
    """
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")

    fake_arm = MagicMock()
    fake_ctrl = MagicMock()
    fake_ctrl._running = False
    fake_gripper = MagicMock()

    with patch("reBotArm_control_py.actuator.RebotArm", return_value=fake_arm), \
         patch(
             "reBotArm_control_py.controllers.RebotArmEndPose",
             return_value=fake_ctrl,
         ), \
         patch(
             "wrc_demo.control.gripper.WrcGripper",
             return_value=fake_gripper,
         ):
        from wrc_demo.control import make_arm

        make_arm(cfg, kinematics=MagicMock())

    fake_gripper.start.assert_called_once()
    fake_ctrl.start.assert_not_called(), (
        "sdk_ctrl.start() must NOT be called when WrcGripper cooperates "
        "with the SDK loop via _loop_cb"
    )


def test_mock_arm_grasp_bridge_is_none():
    """Mock arm has grasp_bridge=None so skills can branch on it uniformly."""
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")

    from wrc_demo.control import make_arm

    arm = make_arm(cfg, kinematics=MagicMock())
    # The MockArm backend doesn't get a bridge; attribute may be absent or None
    assert getattr(arm, "grasp_bridge", None) is None


@pytest.fixture
def sdk_controller():
    from reBotArm_control_py.controllers import RebotArmEndPose

    q = np.array([0.0, 0.1684, 0.6226, -0.4543, 0.0, 0.0])
    arm = SimpleNamespace(
        groups={"arm": SimpleNamespace(num_joints=6)},
        has_gripper=False,
        get_state=lambda: (q, np.zeros(6), np.zeros(6)),
    )
    controller = RebotArmEndPose(arm, arm_control_mode="mit")
    controller._running = True
    return controller


def test_sdk_rejects_wrong_planned_endpoint_before_sending(sdk_controller):
    point = SimpleNamespace(q=sdk_controller.rebotarm.get_state()[0], ik_success=True)
    with patch(
        "reBotArm_control_py.controllers.rebotarm_endpose_controller.track_trajectory",
        return_value=[point],
    ):
        assert not sdk_controller.move_to_traj(0.3, 0.0, 0.3, pitch=0.7)

    assert "trajectory endpoint rejected" in sdk_controller.motion_error
    assert sdk_controller._send_thread is None


def test_sdk_reports_sender_abort_and_recovers_on_next_motion(sdk_controller):
    q = np.full(6, 0.1)
    assert sdk_controller.move_joints_traj([q], 0.01, waypoint_hook=lambda _: False)
    assert not sdk_controller.wait_motion_done(timeout=1.0)
    assert "waypoint 1/1 was rejected" in sdk_controller.motion_error

    assert sdk_controller.move_joints_traj([q], 0.01)
    assert sdk_controller.wait_motion_done(timeout=1.0)
    assert sdk_controller.motion_error is None
    np.testing.assert_array_equal(sdk_controller.get_commanded_joints(), q)


def test_sdk_rs_gravity_feedforward_matches_reference_at_pregrasp(sdk_controller):
    q = np.array([0.68724, 1.99002, 2.02146, -1.48877, 0.15302, 0.65503])
    group = sdk_controller._arm_group
    group.get_positions = MagicMock(return_value=q)
    group._mit_kp = np.array([50.0, 150.0, 150.0, 50.0, 50.0, 50.0])
    group._mit_kd = np.array([3.0, 10.0, 10.0, 5.0, 4.0, 4.0])
    group.send_mit = MagicMock()

    sdk_controller._loop_cb(sdk_controller.rebotarm, dt=0.002)

    # Gravity torques from the rebot_grasp RS URDF at this recorded pose.
    group.send_mit.assert_called_once()
    torque = group.send_mit.call_args.kwargs["tau"]
    np.testing.assert_allclose(torque[1:3], [-8.57182, 5.14226], atol=1e-5)
