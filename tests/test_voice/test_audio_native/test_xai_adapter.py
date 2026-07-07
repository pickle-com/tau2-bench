from tau2.voice.audio_native.tick_result import TickResult
from tau2.voice.audio_native.xai.discrete_time_adapter import DiscreteTimeXAIAdapter
from tau2.voice.audio_native.xai.events import XAIInputTranscriptionCompletedEvent


def test_xai_input_transcription_is_preserved_in_tick_result() -> None:
    adapter = DiscreteTimeXAIAdapter(tick_duration_ms=200)
    result = TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0,
        bytes_per_tick=1600,
        bytes_per_second=8000,
    )

    adapter._process_event(
        result,
        XAIInputTranscriptionCompletedEvent(
            item_id="item_123",
            transcript="Yusuf Rossi, zip code one nine one two two.",
        ),
    )

    assert result.input_audio_transcripts == [
        {
            "item_id": "item_123",
            "content_index": None,
            "transcript": "Yusuf Rossi, zip code one nine one two two.",
        }
    ]
