import os

from tau2.scripts.check_voice_run_readiness import build_readiness_report


def _clear_env(monkeypatch):
    keys = [
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "ELEVENLABS_API_KEY",
        "DEEPGRAM_API_KEY",
        "TAU2_VOICE_ID_MATT_DELANEY",
        "TAU2_VOICE_ID_LISA_BRENNER",
        "TAU2_VOICE_ID_MILDRED_KAPLAN",
        "TAU2_VOICE_ID_ARJUN_ROY",
        "TAU2_VOICE_ID_WEI_LIN",
        "TAU2_VOICE_ID_MAMADOU_DIALLO",
        "TAU2_VOICE_ID_PRIYA_PATIL",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_openai_control_readiness_reports_missing_keys(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    report = build_readiness_report(
        provider="openai",
        speech_complexity="control",
        tool_mentor=True,
        tool_mentor_model="gpt-4.1-2025-04-14",
        require_voice_ids=True,
    )

    missing_names = {item["name"] for item in report["missing"]}
    assert report["ok"] is False
    assert missing_names == {
        "ELEVENLABS_API_KEY",
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "TAU2_VOICE_ID_MATT_DELANEY",
        "TAU2_VOICE_ID_LISA_BRENNER",
    }
    openai_items = [
        item for item in report["missing"] if item["name"] == "OPENAI_API_KEY"
    ]
    assert len(openai_items) == 1
    assert "tool mentor LLM" in openai_items[0]["reason"]


def test_xai_control_readiness_passes_with_required_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram")
    monkeypatch.setenv("TAU2_VOICE_ID_MATT_DELANEY", "matt")
    monkeypatch.setenv("TAU2_VOICE_ID_LISA_BRENNER", "lisa")

    report = build_readiness_report(
        provider="xai",
        speech_complexity="control",
        tool_mentor=True,
        tool_mentor_model="xai/grok-voice-think-fast-1.0",
        require_voice_ids=True,
    )

    assert report["ok"] is True
    assert report["missing"] == []


def test_readiness_loads_cwd_dotenv(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "XAI_API_KEY=xai",
                "ELEVENLABS_API_KEY=eleven",
                "DEEPGRAM_API_KEY=deepgram",
                "TAU2_VOICE_ID_MATT_DELANEY=matt",
                "TAU2_VOICE_ID_LISA_BRENNER=lisa",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    report = build_readiness_report(
        provider="xai",
        speech_complexity="control",
        tool_mentor=True,
        tool_mentor_model="xai/grok-voice-think-fast-1.0",
        require_voice_ids=True,
    )

    assert report["ok"] is True
    assert report["missing"] == []


def test_readiness_dotenv_does_not_override_exported_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "exported")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram")
    (tmp_path / ".env").write_text("XAI_API_KEY=dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    report = build_readiness_report(
        provider="xai",
        speech_complexity="control",
        tool_mentor=True,
        tool_mentor_model="xai/grok-voice-think-fast-1.0",
        require_voice_ids=False,
    )

    assert report["ok"] is True
    assert report["missing"] == []
    assert report["requirements"][2]["name"] == "XAI_API_KEY"
    assert os.environ["XAI_API_KEY"] == "exported"


def test_gemini_accepts_google_alternative_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram")

    report = build_readiness_report(
        provider="gemini",
        speech_complexity="regular",
        tool_mentor=False,
        tool_mentor_model="gpt-4.1-2025-04-14",
        require_voice_ids=False,
    )

    assert report["ok"] is True
    assert report["missing"] == []


def test_gemini_tool_mentor_reports_provider_compatibility_issue(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram")

    report = build_readiness_report(
        provider="gemini",
        speech_complexity="regular",
        tool_mentor=True,
        tool_mentor_model="gemini/gemini-3.1-flash",
        require_voice_ids=False,
    )

    assert report["ok"] is False
    assert report["missing"] == []
    assert report["compatibility_issues"][0]["code"] == (
        "tool_mentor_provider_unsupported"
    )
    assert report["compatibility_issues"][0]["supported_providers"] == [
        "openai",
        "xai",
    ]
