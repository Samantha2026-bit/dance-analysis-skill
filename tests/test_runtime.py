from __future__ import annotations

from dance_pose_coach.runtime import inspect_runtime, lan_address


def test_current_runtime_has_a_stable_shape():
    info = inspect_runtime()
    assert info.system
    assert info.machine
    assert info.python.count(".") >= 1
    assert isinstance(info.supported, bool)


def test_windows_x64_is_supported(monkeypatch):
    monkeypatch.setattr("dance_pose_coach.runtime.platform.system", lambda: "Windows")
    monkeypatch.setattr("dance_pose_coach.runtime.platform.machine", lambda: "AMD64")
    assert inspect_runtime().supported is True


def test_windows_arm_is_reported_as_unverified(monkeypatch):
    monkeypatch.setattr("dance_pose_coach.runtime.platform.system", lambda: "Windows")
    monkeypatch.setattr("dance_pose_coach.runtime.platform.machine", lambda: "ARM64")
    info = inspect_runtime()
    assert info.supported is False
    assert "未验证" in info.message


def test_lan_address_contains_requested_port(monkeypatch):
    class FakeSocket:
        def connect(self, _address):
            return None

        def getsockname(self):
            return ("192.168.1.23", 12345)

        def close(self):
            return None

    monkeypatch.setattr("dance_pose_coach.runtime.socket.socket", lambda *_args: FakeSocket())
    assert lan_address(8765) == "http://192.168.1.23:8765"
