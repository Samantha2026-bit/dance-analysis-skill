from types import SimpleNamespace

from dance_pose_coach.app import _analyze_ui, build_parser, launch_app


def test_app_defaults_to_loopback_mode():
    args = build_parser().parse_args([])
    assert args.mode == "local"
    assert args.port == 7860


def test_phone_mode_is_explicit():
    args = build_parser().parse_args(["--mode", "lan", "--password", "secret"])
    assert args.mode == "lan"
    assert args.password == "secret"


class _FakeApp:
    def __init__(self):
        self.launch_options = None

    def queue(self, **_options):
        return self

    def launch(self, **options):
        self.launch_options = options


def _supported_runtime():
    return SimpleNamespace(supported=True)


def test_local_mode_binds_only_to_loopback(monkeypatch):
    fake = _FakeApp()
    monkeypatch.setattr("dance_pose_coach.app.inspect_runtime", _supported_runtime)
    monkeypatch.setattr("dance_pose_coach.app.build_app", lambda: fake)
    launch_app("local", 7860, None, False)
    assert fake.launch_options["server_name"] == "127.0.0.1"
    assert fake.launch_options["share"] is False
    assert "auth" not in fake.launch_options


def test_lan_mode_is_password_protected_and_never_public(monkeypatch):
    fake = _FakeApp()
    monkeypatch.setattr("dance_pose_coach.app.inspect_runtime", _supported_runtime)
    monkeypatch.setattr("dance_pose_coach.app.build_app", lambda: fake)
    monkeypatch.setattr("dance_pose_coach.app.lan_address", lambda port: f"http://lan:{port}")
    launch_app("lan", 8765, "secret", False)
    assert fake.launch_options["server_name"] == "0.0.0.0"
    assert fake.launch_options["auth"] == ("dance", "secret")
    assert fake.launch_options["share"] is False


def test_ui_uses_solo_mode_when_reference_is_omitted(monkeypatch, tmp_path):
    report = tmp_path / "report.html"
    raw = tmp_path / "report.json"
    report.write_text("solo", encoding="utf-8")
    raw.write_text("{}", encoding="utf-8")
    result = SimpleNamespace(pose_confidence=0.8)
    monkeypatch.setattr(
        "dance_pose_coach.app.analyze_solo",
        lambda **_kwargs: (result, {"html": report, "json": raw}),
    )
    monkeypatch.setattr("dance_pose_coach.app.render_solo_html", lambda _result: "solo html")
    rendered, html_path, json_path, status = _analyze_ui(
        "video.mp4",
        None,
        "",
        "",
        "自动判断",
        6,
        120,
        False,
        progress=lambda *_args, **_kwargs: None,
    )
    assert rendered == "solo html"
    assert html_path == str(report)
    assert json_path == str(raw)
    assert "单视频分析完成" in status
