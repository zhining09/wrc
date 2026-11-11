import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wrc_demo import config
from wrc_demo.apps import demo
from wrc_demo.perception.camera_base import make_camera
from wrc_demo.scripts import calib_top_orbbec as calib


@pytest.fixture
def camera_config(tmp_path, monkeypatch):
    directory = tmp_path / "configs"
    shutil.copytree(config.CONFIG_DIR, directory)
    monkeypatch.setattr(config, "CONFIG_DIR", directory)
    monkeypatch.setattr(calib, "CALIB_DIR", tmp_path / "calibration")
    path = directory / "demo.yaml"
    contents = yaml.safe_load(path.read_text())
    contents["camera"] = {"profile": "d435i"}
    path.write_text(yaml.safe_dump(contents))
    for name in ("d435i", "d405"):
        profile = directory / "cameras" / f"{name}.yaml"
        contents = yaml.safe_load(profile.read_text())
        contents["extrinsics"]["source"] = str(tmp_path / f"{name}.json")
        profile.write_text(yaml.safe_dump(contents))


def test_real_demo_uses_yaml_camera_and_cli_overrides(camera_config):
    assert config.load_demo_config().camera.name == "mock"
    assert config.load_demo_config(arm="rebot_rs").camera.name == "d435i"
    cfg = config.load_demo_config(camera="orbbec_overhead", arm="rebot_rs")
    assert cfg.camera.name == "orbbec_overhead"
    cfg = config.load_demo_config(camera="orbbec_overhead", cameras=["d435i", "mock"], arm="rebot_rs")
    assert [camera["name"] for camera in cfg.cameras] == ["d435i", "mock"]


@pytest.mark.parametrize(("profile", "model"), [("d435i", "D435i"), ("d405", "D405")])
def test_realsense_calibration_uses_selected_profile(camera_config, monkeypatch, capsys, profile, model):
    path = config.CONFIG_DIR / "demo.yaml"
    contents = yaml.safe_load(path.read_text())
    contents["camera"]["profile"] = profile
    path.write_text(yaml.safe_dump(contents))
    serial = f"{model}-TEST"

    def list_device(requested_model):
        assert requested_model == model
        return [{"index": 0, "name": f"Intel RealSense {model}", "serial": serial}]

    monkeypatch.setattr(calib, "list_realsense_devices", list_device)
    assert calib.main(["--list"]) == 0
    assert serial in capsys.readouterr().out
    assert calib.main(["--bind", "top"]) == 0
    cfg = config.load_demo_config(arm="rebot_rs")
    assert cfg.camera.serial == serial
    assert not (calib.CALIB_DIR / "cameras.json").exists()

    def collect(**kwargs):
        assert kwargs["camera_cfg"].type == "realsense"
        assert kwargs["serial"] == cfg.camera.serial
        assert kwargs["out_json"] == Path(cfg.camera.extrinsics.source)
        return calib._synthesize_samples(12)[0], "test"

    monkeypatch.setattr(calib, "_collect_samples", collect)
    assert calib.main([]) == 0
    assert json.loads(Path(cfg.camera.extrinsics.source).read_text())["camera_serial"] == serial


@pytest.mark.parametrize("inverse_distortion", [False, True])
def test_d405_requests_matched_streams_and_filters_depth(monkeypatch, inverse_distortion):
    d405 = {"name": "Intel RealSense D405", "serial": "D405-TEST"}
    d435i = {"name": "Intel RealSense D435I", "serial": "D435I-TEST"}
    device = SimpleNamespace(
        get_info=d405.__getitem__,
        first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.0001),
    )
    settings = SimpleNamespace(streams=[], serial=None)
    settings.enable_stream = lambda *args: settings.streams.append(args)
    settings.enable_device = lambda serial: setattr(settings, "serial", serial)
    intr = SimpleNamespace(
        width=3, height=1, fx=1, fy=1, ppx=0, ppy=0,
        model="inverse_brown_conrady", coeffs=[float(inverse_distortion), 0, 0, 0, 0],
    )
    video = SimpleNamespace(get_intrinsics=lambda: intr)
    stream = SimpleNamespace(as_video_stream_profile=lambda: video)
    profile = SimpleNamespace(get_stream=lambda _: stream, get_device=lambda: device)
    color = np.repeat(np.array([[[10], [20], [30]]], dtype=np.uint8), 3, axis=2)
    frames = SimpleNamespace(
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: color),
        get_depth_frame=lambda: SimpleNamespace(get_data=lambda: np.array([[600, 6000, 6600]])),
    )
    pipeline = SimpleNamespace(start=lambda _: profile, stop=lambda: None, wait_for_frames=lambda _: frames)
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: settings,
        context=lambda: SimpleNamespace(query_devices=lambda: [SimpleNamespace(get_info=d435i.__getitem__), device]),
        camera_info=SimpleNamespace(name="name", serial_number="serial"),
        stream=SimpleNamespace(depth="depth", color="color"),
        format=SimpleNamespace(z16="z16", bgr8="bgr8"),
        distortion=SimpleNamespace(inverse_brown_conrady="inverse_brown_conrady"),
        rs2_project_point_to_pixel=lambda intrinsics, point: (point[0] + 1, point[1]),
        align=lambda _: SimpleNamespace(process=lambda frames: frames),
    ))

    camera_config = config.load_profile("cameras", "d405").as_dict()
    camera_config["serial"] = None
    with make_camera(config.Cfg(camera_config)) as camera:
        assert settings.serial == "D405-TEST"
        assert settings.streams == [("depth", 848, 480, "z16", 15), ("color", 848, 480, "bgr8", 15)]
        assert camera.D is None
        frame = camera.get_frame()
        if inverse_distortion:
            np.testing.assert_array_equal(frame.rgb[0, :, 0], [20, 30, 0])
            np.testing.assert_allclose(frame.depth_m, [[0.6, 0.0, 0.0]])
        else:
            np.testing.assert_array_equal(frame.rgb, color)
            np.testing.assert_allclose(frame.depth_m, [[0.0, 0.6, 0.0]])


def test_explicit_orbbec_binding_preserves_d435i_profile(camera_config, tmp_path):
    assert calib.main(["--camera", "orbbec_overhead", "--bind", "top", "--serial", "OB-TEST"]) == 0
    assert config.load_profile("cameras", "orbbec_overhead").serial == "OB-TEST"
    assert config.load_profile("cameras", "d435i").serial is None
    assert json.loads((tmp_path / "calibration/cameras.json").read_text())["top"]["serial"] == "OB-TEST"


def test_real_grasp_rejects_missing_or_other_camera_calibration(camera_config, monkeypatch, tmp_path):
    def must_not_start(*args, **kwargs):
        pytest.fail("a real arm must stay disconnected until the camera calibration is valid")

    monkeypatch.setattr(demo, "make_arm", must_not_start)
    monkeypatch.setattr(demo, "Kinematics", lambda **kwargs: None)
    cfg = config.load_demo_config(arm="rebot_rs")
    with pytest.raises(ValueError, match="needs a hand-eye calibration"):
        demo.build_runtime(cfg, tmp_path)

    assert calib.main(["--bind", "top", "--serial", "NEW-SERIAL"]) == 0
    (tmp_path / "d435i.json").write_text('{"camera_serial": "OLD-SERIAL"}')
    cfg = config.load_demo_config(arm="rebot_rs")
    with pytest.raises(ValueError, match="calibrated for 'OLD-SERIAL'"):
        demo.build_runtime(cfg, tmp_path)
