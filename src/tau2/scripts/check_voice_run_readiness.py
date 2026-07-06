#!/usr/bin/env python3
"""Preflight checks for τ-Voice benchmark runs."""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from tau2.data_model.voice_personas import CONTROL_PERSONA_NAMES, REGULAR_PERSONA_NAMES

Provider = Literal["openai", "xai", "gemini", "livekit"]
SpeechComplexity = Literal["control", "regular"]
TOOL_MENTOR_SUPPORTED_PROVIDERS = {"openai", "xai"}


@dataclass(frozen=True)
class Requirement:
    name: str
    reason: str
    alternatives: tuple[str, ...] = ()


def _has_any_env(names: tuple[str, ...]) -> bool:
    return any(bool(os.environ.get(name)) for name in names)


def _load_local_dotenv() -> None:
    """Load a cwd-local .env file without overriding exported env vars."""
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def _env_status(requirement: Requirement) -> dict[str, Any]:
    names = (requirement.name, *requirement.alternatives)
    return {
        "name": requirement.name,
        "alternatives": list(requirement.alternatives),
        "reason": requirement.reason,
        "present": _has_any_env(names),
    }


def _provider_requirements(provider: Provider) -> list[Requirement]:
    if provider == "openai":
        return [
            Requirement("OPENAI_API_KEY", "OpenAI realtime agent provider"),
        ]
    if provider == "xai":
        return [
            Requirement("XAI_API_KEY", "xAI realtime agent provider"),
        ]
    if provider == "gemini":
        return [
            Requirement(
                "GEMINI_API_KEY",
                "Gemini Live agent provider",
                alternatives=("GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"),
            ),
        ]
    if provider == "livekit":
        return [
            Requirement("DEEPGRAM_API_KEY", "LiveKit cascaded STT/TTS"),
            Requirement("OPENAI_API_KEY", "LiveKit cascaded default LLM"),
        ]
    raise ValueError(f"Unsupported provider: {provider}")


def _mentor_requirements(enabled: bool, model: str) -> list[Requirement]:
    if not enabled:
        return []
    if model.startswith("xai/") or model.startswith("grok"):
        return [Requirement("XAI_API_KEY", "tool mentor LLM")]
    if model.startswith("gemini/") or model.startswith("google/"):
        return [
            Requirement(
                "GEMINI_API_KEY",
                "tool mentor LLM",
                alternatives=("GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"),
            )
        ]
    return [Requirement("OPENAI_API_KEY", "tool mentor LLM")]


def _voice_persona_requirements(
    speech_complexity: SpeechComplexity,
) -> list[Requirement]:
    names = (
        CONTROL_PERSONA_NAMES
        if speech_complexity == "control"
        else CONTROL_PERSONA_NAMES + REGULAR_PERSONA_NAMES
    )
    return [
        Requirement(
            f"TAU2_VOICE_ID_{name.upper()}",
            f"local ElevenLabs voice persona for {speech_complexity} speech",
        )
        for name in names
    ]


def _dedupe_requirements(requirements: list[Requirement]) -> list[Requirement]:
    """Merge duplicate env requirements while preserving first-seen order."""
    merged: dict[tuple[str, tuple[str, ...]], Requirement] = {}
    for requirement in requirements:
        key = (requirement.name, requirement.alternatives)
        existing = merged.get(key)
        if existing is None:
            merged[key] = requirement
        elif requirement.reason not in existing.reason:
            merged[key] = Requirement(
                name=existing.name,
                alternatives=existing.alternatives,
                reason=f"{existing.reason}; {requirement.reason}",
            )
    return list(merged.values())


def build_readiness_report(
    provider: Provider,
    speech_complexity: SpeechComplexity,
    tool_mentor: bool,
    tool_mentor_model: str,
    require_voice_ids: bool,
) -> dict[str, Any]:
    """Return a structured readiness report without printing secrets."""
    _load_local_dotenv()
    requirements = [
        Requirement("ELEVENLABS_API_KEY", "voice user simulator TTS"),
        Requirement("DEEPGRAM_API_KEY", "voice user simulator transcription"),
        *_provider_requirements(provider),
        *_mentor_requirements(tool_mentor, tool_mentor_model),
    ]
    if require_voice_ids:
        requirements.extend(_voice_persona_requirements(speech_complexity))

    requirements = _dedupe_requirements(requirements)
    statuses = [_env_status(requirement) for requirement in requirements]
    missing = [item for item in statuses if not item["present"]]
    compatibility_issues = []
    if tool_mentor and provider not in TOOL_MENTOR_SUPPORTED_PROVIDERS:
        compatibility_issues.append(
            {
                "code": "tool_mentor_provider_unsupported",
                "message": (
                    "Tool mentor synthetic context delivery is currently "
                    "implemented for openai and xai audio-native providers."
                ),
                "provider": provider,
                "supported_providers": sorted(TOOL_MENTOR_SUPPORTED_PROVIDERS),
            }
        )

    return {
        "provider": provider,
        "speech_complexity": speech_complexity,
        "tool_mentor": tool_mentor,
        "tool_mentor_model": tool_mentor_model if tool_mentor else None,
        "require_voice_ids": require_voice_ids,
        "ok": not missing and not compatibility_issues,
        "requirements": statuses,
        "missing": missing,
        "compatibility_issues": compatibility_issues,
    }


def print_text_report(report: dict[str, Any]) -> None:
    """Print a compact terminal report."""
    print("Voice Run Readiness")
    print(f"  provider: {report['provider']}")
    print(f"  speech_complexity: {report['speech_complexity']}")
    print(f"  tool_mentor: {report['tool_mentor']}")
    print(f"  ok: {report['ok']}")
    print("")
    for issue in report["compatibility_issues"]:
        print(f"  [unsupported] {issue['code']} - {issue['message']}")
    for item in report["requirements"]:
        label = "ok" if item["present"] else "missing"
        names = [item["name"], *item["alternatives"]]
        print(f"  [{label}] {' or '.join(names)} - {item['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check τ-Voice run readiness.")
    parser.add_argument(
        "--provider",
        choices=("openai", "xai", "gemini", "livekit"),
        default="openai",
    )
    parser.add_argument(
        "--speech-complexity",
        choices=("control", "regular"),
        default="regular",
    )
    parser.add_argument("--tool-mentor", action="store_true")
    parser.add_argument("--tool-mentor-model", default="gpt-5.5")
    parser.add_argument(
        "--skip-voice-id-check",
        action="store_true",
        help="Skip local TAU2_VOICE_ID_* checks for Sierra-managed final runs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    report = build_readiness_report(
        provider=args.provider,
        speech_complexity=args.speech_complexity,
        tool_mentor=args.tool_mentor,
        tool_mentor_model=args.tool_mentor_model,
        require_voice_ids=not args.skip_voice_id_check,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
