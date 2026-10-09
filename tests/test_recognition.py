from dance_pose_coach.recognition import query_from_filename, recognize_track


def test_filename_fallback_removes_common_noise():
    assert query_from_filename("/tmp/Artist_Song_dance_practice_cover.mp4") == "Artist Song"


def test_recognition_is_offline_by_default(monkeypatch):
    def unexpected_online_call(*_args, **_kwargs):
        raise AssertionError("online recognition must be opt-in")

    monkeypatch.setattr("dance_pose_coach.recognition.asyncio.run", unexpected_online_call)
    result = recognize_track("unused.wav", "Artist_Song_dance.mp4")
    assert result.method == "filename"
    assert result.title == "Artist Song"
