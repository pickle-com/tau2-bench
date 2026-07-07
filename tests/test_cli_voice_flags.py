"""CLI parser smoke test for the voice + tool-mentor flag surface.

Pins that the exact command shape used for tool-mentor voice runs parses:
every --tool-mentor-* and --xai-vad-* flag is accepted and lands on the
expected argparse destination with the expected type.
"""

import argparse

from tau2.cli import add_run_args

CANONICAL_RUN_ARGS = [
    "--domain",
    "telecom",
    "--audio-native",
    "--audio-native-provider",
    "xai",
    "--audio-native-model",
    "grok-voice-think-fast-1.0",
    "--xai-vad-threshold",
    "0.1",
    "--xai-vad-silence-duration-ms",
    "1200",
    "--xai-vad-prefix-padding-ms",
    "600",
    "--num-trials",
    "1",
    "--max-steps",
    "6000",
    "--max-concurrency",
    "10",
    "--user",
    "voice_streaming_user_simulator",
    "--user-llm",
    "gpt-5.5-2026-04-23",
    "--user-llm-args",
    '{"reasoning_effort": "xhigh"}',
    "--tool-mentor",
    "--tool-mentor-model",
    "gemini/gemini-3.5-flash",
    "--tool-mentor-reasoning-effort",
    "high",
    "--tool-mentor-read-timeout",
    "10.0",
    "--tool-mentor-write-timeout",
    "15.0",
    "--tool-mentor-realtime-wait",
    "--tool-mentor-realtime-workers",
    "5",
    "--speech-complexity",
    "regular",
    "--skip-voice-id-check",
    "--auto-resume",
]


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    return parser.parse_args(argv)


def test_canonical_mentor_voice_command_parses():
    args = _parse(CANONICAL_RUN_ARGS)

    assert args.audio_native is True
    assert args.audio_native_provider == "xai"
    assert args.audio_native_model == "grok-voice-think-fast-1.0"

    assert args.xai_vad_threshold == 0.1
    assert args.xai_vad_silence_duration_ms == 1200
    assert args.xai_vad_prefix_padding_ms == 600
    assert args.xai_vad_idle_timeout_ms is None
    assert args.xai_audio_format == "pcmu"

    assert args.tool_mentor is True
    assert args.tool_mentor_model == "gemini/gemini-3.5-flash"
    assert args.tool_mentor_reasoning_effort == "high"
    assert args.tool_mentor_read_timeout == 10.0
    assert args.tool_mentor_write_timeout == 15.0
    assert args.tool_mentor_realtime_wait is True
    assert args.tool_mentor_realtime_workers == 5

    assert args.user_llm_args == {"reasoning_effort": "xhigh"}
    assert args.skip_voice_id_check is True


def test_mentor_flags_default_off():
    args = _parse(["--domain", "telecom"])

    assert args.tool_mentor is False
    assert args.tool_mentor_stop_interception is None
    assert args.tool_mentor_escalation_tools is None
    assert args.preflight_only is False
    assert args.xai_vad_threshold is None
    assert args.llm_log_mode is None


def test_stop_interception_and_escalation_flags_parse():
    args = _parse(
        [
            "--domain",
            "telecom",
            "--tool-mentor",
            "--no-tool-mentor-stop-interception",
            "--tool-mentor-escalation-tools",
        ]
    )

    assert args.tool_mentor_stop_interception is False
    assert args.tool_mentor_escalation_tools == []
