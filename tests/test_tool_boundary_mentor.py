import asyncio
import json
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from types import SimpleNamespace

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import AudioNativeConfig, TerminationReason
from tau2.data_model.tasks import (
    Action,
    EnvAssertion,
    EvaluationCriteria,
    RewardType,
    Task,
    UserScenario,
)
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.evaluator.evaluator_action import FullDuplexActionEvaluator
from tau2.evaluator.evaluator_env import FullDuplexEnvironmentEvaluator
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.utils.llm_utils import set_llm_log_dir, set_llm_log_mode
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.mentor import tool_boundary_mentor as mentor_module
from tau2.voice.audio_native.mentor.tool_boundary_mentor import (
    MentorDecision,
    MentorEvent,
    ToolBoundaryMentor,
)
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import (
    InputAudioTranscriptionCompletedEvent,
)
from tau2.voice.audio_native.tick_result import TickResult


def _history_payload_from_message(message: dict) -> dict:
    content = message["content"]
    assert content.startswith("<history_block>\n")
    assert "\n</history_block>" in content
    raw = content.split("<history_block>\n", 1)[1].split("\n</history_block>", 1)[0]
    return json.loads(raw)


def _current_tool_event_from_message(message: dict) -> dict:
    content = message["content"]
    assert "<current_tool_event>\n" in content
    assert content.endswith("\n</current_tool_event>")
    return json.loads(
        content.split("<current_tool_event>\n", 1)[1].removesuffix(
            "\n</current_tool_event>"
        )
    )


class _CapturingAdapter(DiscreteTimeAdapter):
    def __init__(self):
        super().__init__(tick_duration_ms=100)
        self.tool_result_batches = []
        self.synthetic_context_batches = []

    def connect(
        self, system_prompt: str, tools: list[Tool], vad_config, modality="audio"
    ):
        return None

    def disconnect(self):
        return None

    @property
    def is_connected(self) -> bool:
        return True

    def run_tick(self, user_audio: bytes, tick_number: int | None = None) -> TickResult:
        self.tool_result_batches.append(list(self._pending_tool_results))
        self.synthetic_context_batches.append(
            list(self._pending_synthetic_agent_contexts)
        )
        self._pending_tool_results.clear()
        self._pending_synthetic_agent_contexts.clear()
        return TickResult(
            tick_number=tick_number or 0,
            audio_sent_bytes=len(user_audio),
            audio_sent_duration_ms=0,
            user_audio_data=user_audio,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=self.audio_format.bytes_per_second,
        )

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        return None

    async def _flush_pending_tool_results(self) -> None:
        return None


class _MentorToolkit(ToolKitBase):
    def __init__(self):
        super().__init__()
        self.cancelled = False
        self.write_log = []

    @is_tool(ToolType.READ, mutates_state=False)
    def get_product_details(self, product_id: str):
        return {
            "product_id": product_id,
            "variants": {
                "v1": {"available": True},
                "v2": {"available": False},
                "v3": {"available": True},
            },
        }

    @is_tool(ToolType.WRITE, mutates_state=True)
    def cancel_order(self, order_id: str):
        self.cancelled = True
        return {"order_id": order_id, "cancelled": True}

    @is_tool(ToolType.WRITE, mutates_state=True)
    def record_write(self, value: str):
        self.write_log.append(value)
        return {"value": value, "write_log": list(self.write_log)}

    def assert_cancelled(self, expected: bool):
        return self.cancelled is expected

    def get_db_hash(self) -> str:
        return f"cancelled:{self.cancelled}"


class _GenericMutatingToolkit(_MentorToolkit):
    @is_tool(ToolType.GENERIC, mutates_state=True)
    def apply_stateful_utility(self, value: str):
        self.cancelled = value == "cancel"
        return {"value": value}


class _FakeAgent:
    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        return (
            AssistantMessage(
                role="assistant",
                content=None,
                contains_speech=False,
            ),
            state,
        )

    def get_init_state(self, message_history=None):
        return {}

    def set_seed(self, seed: int) -> None:
        return None

    @staticmethod
    def is_stop(message) -> bool:
        return False

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


class _ScriptedReadAgent:
    def __init__(self):
        self.received = []

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
        synthetic_agent_context=None,
    ):
        self.received.append(
            {
                "tool_results": tool_results,
                "synthetic_agent_context": synthetic_agent_context,
            }
        )
        step = state.get("step", 0)
        state["step"] = step + 1
        if step == 0:
            return (
                AssistantMessage(
                    role="assistant",
                    content=None,
                    contains_speech=False,
                    tool_calls=[
                        ToolCall(
                            id="call_read",
                            name="get_product_details",
                            arguments={"product_id": "p1"},
                        )
                    ],
                ),
                state,
            )
        return (
            AssistantMessage(
                role="assistant",
                content="I see two available variants.",
                contains_speech=True,
            ),
            state,
        )

    def get_init_state(self, message_history=None):
        return {"step": 0}

    def set_seed(self, seed: int) -> None:
        return None

    @staticmethod
    def is_stop(message) -> bool:
        return False

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


class _ScriptedTwoReadAgent:
    def __init__(self):
        self.received = []

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
        synthetic_agent_context=None,
    ):
        self.received.append(
            {
                "tool_results": tool_results,
                "synthetic_agent_context": synthetic_agent_context,
            }
        )
        step = state.get("step", 0)
        state["step"] = step + 1
        if step == 0:
            return (
                AssistantMessage(
                    role="assistant",
                    content=None,
                    contains_speech=False,
                    tool_calls=[
                        ToolCall(
                            id="call_read_1",
                            name="get_product_details",
                            arguments={"product_id": "p1"},
                        ),
                        ToolCall(
                            id="call_read_2",
                            name="get_product_details",
                            arguments={"product_id": "p2"},
                        ),
                    ],
                ),
                state,
            )
        return (
            AssistantMessage(
                role="assistant",
                content="I have the details.",
                contains_speech=True,
            ),
            state,
        )

    def get_init_state(self, message_history=None):
        return {"step": 0}

    def set_seed(self, seed: int) -> None:
        return None

    @staticmethod
    def is_stop(message) -> bool:
        return False

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


class _ScriptedReadWriteAgent:
    def __init__(self):
        self.received = []

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
        synthetic_agent_context=None,
    ):
        self.received.append(
            {
                "tool_results": tool_results,
                "synthetic_agent_context": synthetic_agent_context,
            }
        )
        step = state.get("step", 0)
        state["step"] = step + 1
        if step == 0:
            return (
                AssistantMessage(
                    role="assistant",
                    content=None,
                    contains_speech=False,
                    tool_calls=[
                        ToolCall(
                            id="call_read",
                            name="get_product_details",
                            arguments={"product_id": "p1"},
                        ),
                        ToolCall(
                            id="call_write",
                            name="cancel_order",
                            arguments={"order_id": "order_1"},
                        ),
                    ],
                ),
                state,
            )
        return (
            AssistantMessage(
                role="assistant",
                content="Done.",
                contains_speech=True,
            ),
            state,
        )

    def get_init_state(self, message_history=None):
        return {"step": 0}

    def set_seed(self, seed: int) -> None:
        return None

    @staticmethod
    def is_stop(message) -> bool:
        return False

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


class _ScriptedTwoWriteAgent:
    def __init__(self):
        self.received = []

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
        synthetic_agent_context=None,
    ):
        self.received.append(
            {
                "tool_results": tool_results,
                "synthetic_agent_context": synthetic_agent_context,
            }
        )
        step = state.get("step", 0)
        state["step"] = step + 1
        if step == 0:
            return (
                AssistantMessage(
                    role="assistant",
                    content=None,
                    contains_speech=False,
                    tool_calls=[
                        ToolCall(
                            id="call_write_1",
                            name="record_write",
                            arguments={"value": "first"},
                        ),
                        ToolCall(
                            id="call_write_2",
                            name="record_write",
                            arguments={"value": "second"},
                        ),
                    ],
                ),
                state,
            )
        return (
            AssistantMessage(
                role="assistant",
                content="Done.",
                contains_speech=True,
            ),
            state,
        )

    def get_init_state(self, message_history=None):
        return {"step": 0}

    def set_seed(self, seed: int) -> None:
        return None

    @staticmethod
    def is_stop(message) -> bool:
        return False

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


class _ScriptedWriteAgent:
    def __init__(self):
        self.received = []

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
        synthetic_agent_context=None,
    ):
        self.received.append(
            {
                "tool_results": tool_results,
                "synthetic_agent_context": synthetic_agent_context,
            }
        )
        step = state.get("step", 0)
        state["step"] = step + 1
        if step == 0:
            return (
                AssistantMessage(
                    role="assistant",
                    content=None,
                    contains_speech=False,
                    tool_calls=[
                        ToolCall(
                            id="call_write",
                            name="cancel_order",
                            arguments={"order_id": "order_1"},
                        )
                    ],
                ),
                state,
            )
        return (
            AssistantMessage(
                role="assistant",
                content="Let me confirm that before changing anything.",
                contains_speech=True,
            ),
            state,
        )

    def get_init_state(self, message_history=None):
        return {"step": 0}

    def set_seed(self, seed: int) -> None:
        return None

    @staticmethod
    def is_stop(message) -> bool:
        return False

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


class _FakeUser:
    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        return (
            UserMessage(
                role="user",
                content=None,
                contains_speech=False,
            ),
            state,
        )

    def get_init_state(self, message_history=None):
        return {}

    def set_seed(self, seed: int) -> None:
        return None

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


def _orchestrator(toolkit: _MentorToolkit) -> FullDuplexOrchestrator:
    env = Environment(
        domain_name="retail",
        policy="Ask for explicit confirmation before cancellation.",
        tools=toolkit,
    )
    task = Task(
        id="mentor_test",
        user_scenario=UserScenario(instructions="Test user scenario."),
    )
    config = AudioNativeConfig(
        provider="xai",
        tool_mentor_enabled=True,
        tool_mentor_mode="heuristic",
    )
    return FullDuplexOrchestrator(
        domain="retail",
        agent=_FakeAgent(),
        user=_FakeUser(),
        environment=env,
        task=task,
        tool_mentor_config=config,
    )


def _orchestrator_with_agent(
    toolkit: _MentorToolkit,
    agent,
    *,
    realtime_wait: bool = False,
    tool_workers: int = 5,
) -> FullDuplexOrchestrator:
    env = Environment(
        domain_name="retail",
        policy="Ask for explicit confirmation before cancellation.",
        tools=toolkit,
    )
    task = Task(
        id="mentor_test",
        user_scenario=UserScenario(instructions="Test user scenario."),
    )
    config = AudioNativeConfig(
        provider="xai",
        tool_mentor_enabled=True,
        tool_mentor_mode="heuristic",
        tool_mentor_realtime_wait=realtime_wait,
        tool_mentor_realtime_workers=tool_workers,
    )
    return FullDuplexOrchestrator(
        domain="retail",
        agent=agent,
        user=_FakeUser(),
        environment=env,
        task=task,
        tool_mentor_config=config,
    )


def test_read_tool_result_stays_official_and_provider_result_gets_note():
    orch = _orchestrator(_MentorToolkit())
    tool_call = ToolCall(
        id="call_read",
        name="get_product_details",
        arguments={"product_id": "p1"},
    )

    executed_calls, results, provider_results, contexts = (
        orch._execute_agent_tool_calls_with_mentor([tool_call])
    )

    assert executed_calls == [tool_call]
    assert len(results) == 1
    assert len(provider_results) == 1
    assert provider_results[0] != results[0]
    official = results[0].content or ""
    assert "mentor_note" not in official
    assert json.loads(official)["variants"]["v2"]["available"] == "False"
    provider_visible = json.loads(provider_results[0].content or "{}")
    assert list(provider_visible.keys())[0] == "mentor_note"
    assert "Available variants: 2" in provider_visible["mentor_note"]
    assert provider_visible["variants"]["v2"]["available"] == "False"
    trace_entry = orch._voice_timing_trace["provider_tool_results"][0]
    assert trace_entry["kind"] == "mentor_augmented_read_result"
    assert trace_entry["official_result_present"] is True
    assert json.loads(trace_entry["content"]) == provider_visible
    provider_event = orch._voice_timing_trace["provider_visible_events"][0]
    assert provider_event["type"] == "tool_result"
    assert provider_event["kind"] == "mentor_augmented_read_result"
    assert provider_event["content"] == trace_entry["content"]
    assert contexts == []


def test_write_gate_block_skips_environment_mutation_and_official_result():
    toolkit = _MentorToolkit()
    orch = _orchestrator(toolkit)
    tool_call = ToolCall(
        id="call_write",
        name="cancel_order",
        arguments={"order_id": "order_1"},
    )

    executed_calls, results, provider_results, contexts = (
        orch._execute_agent_tool_calls_with_mentor([tool_call])
    )

    assert executed_calls == []
    assert results == []
    assert len(provider_results) == 1
    blocked_output = json.loads(provider_results[0].content or "{}")
    assert blocked_output["type"] == "mentor_blocked_tool_call"
    assert provider_results[0].error is False
    assert toolkit.cancelled is False
    trace_entry = orch._voice_timing_trace["provider_tool_results"][0]
    assert trace_entry["kind"] == "mentor_blocked_tool_call"
    assert trace_entry["official_result_present"] is False
    assert json.loads(trace_entry["content"]) == blocked_output
    provider_event = orch._voice_timing_trace["provider_visible_events"][0]
    assert provider_event["type"] == "tool_result"
    assert provider_event["kind"] == "mentor_blocked_tool_call"
    assert provider_event["official_result_present"] is False
    assert contexts == []
    assert orch._voice_timing_trace["tool_events"][0]["skipped_by_mentor"] is True


def test_write_gate_allow_note_augments_provider_result_only(monkeypatch):
    def _allow_with_note(
        self,
        tool_call,
        recent_transcript,
        recent_tool_history=None,
        history_events=None,
    ):
        return (
            MentorDecision(
                decision="allow",
                agent_note="After this write, confirm the changed order status.",
            ),
            MentorEvent(
                stage="pre_write",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_type="write",
                mutates_state=True,
                outcome="allow",
                elapsed_ms=0.0,
                decision="allow",
            ),
            None,
        )

    monkeypatch.setattr(ToolBoundaryMentor, "pre_write_gate", _allow_with_note)
    toolkit = _MentorToolkit()
    orch = _orchestrator(toolkit)
    tool_call = ToolCall(
        id="call_write",
        name="cancel_order",
        arguments={"order_id": "order_1"},
    )

    executed_calls, official_results, provider_results, contexts = (
        orch._execute_agent_tool_calls_with_mentor([tool_call])
    )

    assert executed_calls == [tool_call]
    assert toolkit.cancelled is True
    assert len(official_results) == 1
    assert "mentor_note" not in (official_results[0].content or "")
    provider_payload = json.loads(provider_results[0].content or "{}")
    assert provider_payload["mentor_note"] == (
        "After this write, confirm the changed order status."
    )
    assert provider_payload["cancelled"] == "True"
    trace_entry = orch._voice_timing_trace["provider_tool_results"][0]
    assert trace_entry["kind"] == "mentor_augmented_write_result"
    assert trace_entry["official_result_present"] is True
    assert contexts == []


def test_action_compare_matches_official_name_and_args_behavior():
    action = Action(
        action_id="lookup_1",
        requestor="assistant",
        name="get_product_details",
        arguments={"product_id": "p1"},
    )

    assert action.compare_with_tool_call(
        ToolCall(
            id="call_agent",
            requestor="assistant",
            name="get_product_details",
            arguments={"product_id": "p1"},
        )
    )
    assert action.compare_with_tool_call(
        ToolCall(
            id="call_user",
            requestor="user",
            name="get_product_details",
            arguments={"product_id": "p1"},
        )
    )


def test_tool_mentor_pre_gates_any_mutating_tool_metadata():
    orch = _orchestrator(_GenericMutatingToolkit())

    assert orch.tool_mentor is not None
    assert orch.tool_mentor.should_pre_gate("apply_stateful_utility") is True
    assert orch.tool_mentor.should_post_read_note("apply_stateful_utility") is False


def test_orchestrator_keeps_same_batch_read_result_out_of_write_gate(monkeypatch):
    captured_histories = []

    def _capture_pre_write(
        self,
        tool_call,
        recent_transcript,
        recent_tool_history=None,
        history_events=None,
    ):
        captured_histories.append(list(recent_tool_history or []))
        return (
            MentorDecision(decision="allow"),
            MentorEvent(
                stage="pre_write",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_type="write",
                mutates_state=True,
                outcome="allow",
                elapsed_ms=0.0,
                decision="allow",
            ),
            None,
        )

    monkeypatch.setattr(ToolBoundaryMentor, "pre_write_gate", _capture_pre_write)
    toolkit = _MentorToolkit()
    orch = _orchestrator(toolkit)
    read_call = ToolCall(
        id="call_read",
        name="get_product_details",
        arguments={"product_id": "p1"},
    )
    write_call = ToolCall(
        id="call_write",
        name="cancel_order",
        arguments={"order_id": "order_1"},
    )

    orch._execute_agent_tool_calls_with_mentor([read_call, write_call])

    assert captured_histories
    history = captured_histories[0]
    assert history == []


def test_orchestrator_passes_prior_tick_tool_history_to_write_gate(monkeypatch):
    captured_histories = []

    def _capture_pre_write(
        self,
        tool_call,
        recent_transcript,
        recent_tool_history=None,
        history_events=None,
    ):
        captured_histories.append(list(recent_tool_history or []))
        return (
            MentorDecision(decision="allow"),
            MentorEvent(
                stage="pre_write",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_type="write",
                mutates_state=True,
                outcome="allow",
                elapsed_ms=0.0,
                decision="allow",
            ),
            None,
        )

    monkeypatch.setattr(ToolBoundaryMentor, "pre_write_gate", _capture_pre_write)
    toolkit = _MentorToolkit()
    orch = _orchestrator(toolkit)
    read_call = ToolCall(
        id="call_read",
        name="get_product_details",
        arguments={"product_id": "p1"},
    )
    read_result = orch.environment.get_response(read_call)
    orch.ticks.append(
        Tick(
            tick_id=0,
            timestamp="2026-01-01T00:00:00",
            agent_chunk=AssistantMessage(
                role="assistant",
                content=None,
                contains_speech=False,
            ),
            agent_tool_calls=[read_call],
            agent_tool_results=[read_result],
        )
    )
    write_call = ToolCall(
        id="call_write",
        name="cancel_order",
        arguments={"order_id": "order_1"},
    )

    orch._execute_agent_tool_calls_with_mentor([write_call])

    assert captured_histories
    history = captured_histories[0]
    assert len(history) == 1
    assert history[0]["tool_call_id"] == "call_read"
    assert history[0]["tool_name"] == "get_product_details"
    assert history[0]["arguments"] == {"product_id": "p1"}
    assert "variants" in history[0]["official_result"]


def test_openai_adapter_records_input_audio_transcription_event():
    adapter = DiscreteTimeOpenAIAdapter(
        tick_duration_ms=100,
        provider=SimpleNamespace(),
    )
    result = TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0,
        user_audio_data=b"",
        bytes_per_tick=adapter.bytes_per_tick,
        bytes_per_second=adapter.audio_format.bytes_per_second,
    )
    event = InputAudioTranscriptionCompletedEvent(
        type="conversation.item.input_audio_transcription.completed",
        item_id="item_1",
        content_index=0,
        transcript="Bethan Garcia.",
    )

    asyncio.run(adapter._process_event(result, event))

    assert result.input_audio_transcripts == [
        {
            "item_id": "item_1",
            "content_index": 0,
            "transcript": "Bethan Garcia.",
        }
    ]


def test_recent_transcript_uses_agent_observed_input_transcripts():
    orch = _orchestrator(_MentorToolkit())
    orch.ticks = [
        Tick(
            tick_id=1,
            timestamp="t1",
            user_chunk=UserMessage(
                role="user",
                content="Ethan Garcia.",
                contains_speech=True,
            ),
            agent_chunk=AssistantMessage(
                role="assistant",
                content=None,
                contains_speech=False,
                raw_data={
                    "input_audio_transcripts": [
                        {
                            "item_id": "item_1",
                            "content_index": 0,
                            "transcript": "Bethan Garcia.",
                        },
                        {
                            "item_id": "item_2",
                            "content_index": 0,
                            "transcript": "ZIP 80280.",
                        },
                    ]
                },
            ),
        ),
        Tick(
            tick_id=2,
            timestamp="t2",
            user_chunk=UserMessage(
                role="user",
                content="This gold text should stay hidden.",
                contains_speech=True,
            ),
        ),
        Tick(
            tick_id=3,
            timestamp="t3",
            user_chunk=UserMessage(
                role="user",
                content="Also hidden.",
                contains_speech=True,
            ),
        ),
        Tick(
            tick_id=4,
            timestamp="t4",
            agent_chunk=AssistantMessage(
                role="assistant",
                content="Applying ",
                contains_speech=True,
            ),
        ),
        Tick(
            tick_id=5,
            timestamp="t5",
            agent_chunk=AssistantMessage(
                role="assistant",
                content="the change.",
                contains_speech=True,
            ),
        ),
    ]

    assert orch._recent_transcript() == [
        {"role": "user", "text": "Bethan Garcia. ZIP 80280."},
        {"role": "assistant", "text": "Applying the change."},
    ]


def test_agent_observed_history_uses_provider_visible_tool_result():
    orch = _orchestrator(_MentorToolkit())
    tool_call = ToolCall(
        id="call_read",
        name="get_product_details",
        arguments={"product_id": "p1"},
    )
    official_result = ToolMessage(
        id="call_read",
        requestor="assistant",
        role="tool",
        content='{"variants":{"v1":{"available":true}}}',
    )
    provider_visible = {
        "mentor_note": "Only count variants with available=true.",
        "result": {"variants": {"v1": {"available": True}}},
    }
    orch._voice_timing_trace["provider_tool_results"] = [
        {
            "tool_call_id": "call_read",
            "kind": "mentor_augmented_read_result",
            "content": json.dumps(provider_visible),
        }
    ]
    orch.ticks = [
        Tick(
            tick_id=1,
            timestamp="t1",
            agent_chunk=AssistantMessage(
                role="assistant",
                content="Let me check.",
                contains_speech=True,
                raw_data={
                    "input_audio_transcripts": [
                        {
                            "item_id": "item_1",
                            "content_index": 0,
                            "transcript": "I need the available variant count.",
                        }
                    ]
                },
            ),
            agent_tool_calls=[tool_call],
            agent_tool_results=[official_result],
        )
    ]

    events = orch._recent_agent_observed_history_events()

    assert [event["kind"] for event in events] == [
        "customer_speech_transcript",
        "voice_agent_speech",
        "voice_agent_tool_call",
        "voice_agent_visible_tool_result",
    ]
    assert events[0]["text"] == "I need the available variant count."
    assert events[0]["source"] == "agent_observed_stt_turn_merged"
    assert events[3]["content_source"] == "provider_visible"
    assert json.loads(events[3]["content"]) == provider_visible


def test_agent_observed_history_merges_streaming_chunks_before_tool_boundary():
    orch = _orchestrator(_MentorToolkit())
    tool_call = ToolCall(
        id="call_read",
        name="find_user_id_by_name_zip",
        arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
    )
    tool_result = ToolMessage(
        id="call_read",
        requestor="assistant",
        role="tool",
        content="yusuf_rossi_9620",
    )
    orch.ticks = [
        Tick(
            tick_id=1,
            timestamp="t1",
            agent_chunk=AssistantMessage(
                role="assistant",
                content="Please spell ",
                contains_speech=True,
            ),
        ),
        Tick(
            tick_id=2,
            timestamp="t2",
            agent_chunk=AssistantMessage(
                role="assistant",
                content="your name.",
                contains_speech=True,
            ),
        ),
        Tick(
            tick_id=3,
            timestamp="t3",
            agent_chunk=AssistantMessage(
                role="assistant",
                content=None,
                contains_speech=False,
                raw_data={
                    "input_audio_transcripts": [
                        {"item_id": "item_1", "transcript": "Yusuf."},
                        {"item_id": "item_2", "transcript": "Rossi."},
                        {"item_id": "item_3", "transcript": "ZIP 19122."},
                    ]
                },
            ),
            agent_tool_calls=[tool_call],
            agent_tool_results=[tool_result],
        ),
    ]

    events = orch._recent_agent_observed_history_events()

    assert [event["kind"] for event in events] == [
        "voice_agent_speech",
        "customer_speech_transcript",
        "voice_agent_tool_call",
        "voice_agent_visible_tool_result",
    ]
    assert events[0]["text"] == "Please spell your name."
    assert events[1]["text"] == "Yusuf. Rossi. ZIP 19122."


def test_heuristic_write_gate_uses_user_confirmation_only():
    env = Environment(
        domain_name="retail",
        policy="Ask for explicit confirmation before cancellation. " + ("x" * 5000),
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="xai",
            tool_mentor_enabled=True,
            tool_mentor_mode="heuristic",
        ),
        environment=env,
    )
    decision = mentor._heuristic_pre_write(
        ToolCall(
            id="call_write",
            name="cancel_order",
            arguments={"order_id": "order_1"},
        ),
        recent_transcript=[
            {
                "role": "assistant",
                "text": "Please confirm that you want me to cancel this order.",
            },
            {"role": "user", "text": "What are my options?"},
        ],
    )

    assert decision.decision == "block"


def test_agent_delivers_official_result_then_synthetic_context_to_adapter():
    adapter = _CapturingAdapter()
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="Test policy.",
        adapter=adapter,
        vad_config=object(),
    )
    state = agent.get_init_state()
    tool_result = ToolMessage(
        role="tool",
        id="call_read",
        content='{"variants":{"v1":{"available":"True"}}}',
    )
    synthetic_context = [
        {
            "type": "mentor_note",
            "content": "SYNTHETIC MENTOR NOTE FOR THE ASSISTANT.\nAvailable variants: 1.",
        }
    ]

    agent.get_next_chunk(
        state=state,
        tool_results=tool_result,
        synthetic_agent_context=synthetic_context,
    )

    assert adapter.tool_result_batches == [
        [
            (
                "call_read",
                '{"variants":{"v1":{"available":"True"}}}',
                False,
                False,
            )
        ]
    ]
    assert adapter.synthetic_context_batches == [
        [
            (
                "SYNTHETIC MENTOR NOTE FOR THE ASSISTANT.\nAvailable variants: 1.",
                True,
            )
        ]
    ]


def test_realtime_wait_schedules_agent_tool_result_without_blocking_tick(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def _delayed_execute(self, *, tool_calls, **_kwargs):
        started.set()
        release.wait(timeout=2)
        result = ToolMessage(
            id=tool_calls[0].id,
            role="tool",
            requestor="assistant",
            content='{"ok": true}',
            error=False,
        )
        return tool_calls, [result], [result], []

    monkeypatch.setattr(
        FullDuplexOrchestrator,
        "_execute_agent_tool_calls_with_mentor_context",
        _delayed_execute,
    )
    agent = _ScriptedReadAgent()
    orch = _orchestrator_with_agent(
        _MentorToolkit(),
        agent,
        realtime_wait=True,
    )
    orch.initialize()

    first_step_start = time.perf_counter()
    orch.step()
    first_step_elapsed = time.perf_counter() - first_step_start

    assert started.wait(timeout=1)
    assert first_step_elapsed < 0.2
    first_tool_tick = orch.ticks[-1]
    assert [tool_call.id for tool_call in first_tool_tick.agent_tool_calls] == [
        "call_read"
    ]
    assert first_tool_tick.agent_tool_results == []
    assert agent.received[-1]["tool_results"] is None
    assert orch._pending_agent_tool_jobs

    release.set()
    for _ in range(20):
        if orch._pending_agent_tool_jobs[0].future.done():
            break
        time.sleep(0.01)

    orch.step()

    delivery_tick = orch.ticks[-1]
    assert delivery_tick.agent_tool_results == []
    assert [result.id for result in first_tool_tick.agent_tool_results] == ["call_read"]
    assert agent.received[-1]["tool_results"].id == "call_read"
    assert not orch._pending_agent_tool_jobs
    async_events = orch._voice_timing_trace["async_tool_jobs"]
    assert [event["event"] for event in async_events[:2]] == [
        "scheduled",
        "completed",
    ]
    assert async_events[1]["waited_ticks"] >= 1
    orch._shutdown_agent_tool_executor()


def test_realtime_wait_runs_same_tick_read_tools_in_parallel(monkeypatch):
    release = threading.Event()
    both_started = threading.Event()
    started_ids = []
    lock = threading.Lock()

    def _delayed_execute(self, *, tool_calls, **_kwargs):
        tool_call = tool_calls[0]
        with lock:
            started_ids.append(tool_call.id)
            if len(started_ids) == 2:
                both_started.set()
        release.wait(timeout=2)
        result = ToolMessage(
            id=tool_call.id,
            role="tool",
            requestor="assistant",
            content=f'{{"tool_call_id": "{tool_call.id}"}}',
            error=False,
        )
        return tool_calls, [result], [result], []

    monkeypatch.setattr(
        FullDuplexOrchestrator,
        "_execute_agent_tool_calls_with_mentor_context",
        _delayed_execute,
    )
    agent = _ScriptedTwoReadAgent()
    orch = _orchestrator_with_agent(
        _MentorToolkit(),
        agent,
        realtime_wait=True,
        tool_workers=2,
    )
    orch.initialize()

    orch.step()

    assert both_started.wait(timeout=1)
    scheduled_events = [
        event
        for event in orch._voice_timing_trace["async_tool_jobs"]
        if event["event"] == "scheduled"
    ]
    assert [event["executor"] for event in scheduled_events] == [
        "parallel",
        "parallel",
    ]
    assert {event["worker_count"] for event in scheduled_events} == {2}
    assert sorted(started_ids) == ["call_read_1", "call_read_2"]

    release.set()
    for _ in range(50):
        if all(job.future.done() for job in orch._pending_agent_tool_jobs):
            break
        time.sleep(0.01)

    orch.step()

    assert not orch._pending_agent_tool_jobs
    completed_events = [
        event
        for event in orch._voice_timing_trace["async_tool_jobs"]
        if event["event"] == "completed"
    ]
    assert [event["executor"] for event in completed_events] == [
        "parallel",
        "parallel",
    ]
    orch._shutdown_agent_tool_executor()


def test_realtime_wait_delivers_later_completed_tool_without_head_of_line_block(
    monkeypatch,
):
    release_first = threading.Event()
    second_done = threading.Event()

    def _delayed_first_execute(self, *, tool_calls, **_kwargs):
        tool_call = tool_calls[0]
        if tool_call.id == "call_read_1":
            release_first.wait(timeout=2)
        result = ToolMessage(
            id=tool_call.id,
            role="tool",
            requestor="assistant",
            content=f'{{"tool_call_id": "{tool_call.id}"}}',
            error=False,
        )
        if tool_call.id == "call_read_2":
            second_done.set()
        return tool_calls, [result], [result], []

    monkeypatch.setattr(
        FullDuplexOrchestrator,
        "_execute_agent_tool_calls_with_mentor_context",
        _delayed_first_execute,
    )
    agent = _ScriptedTwoReadAgent()
    orch = _orchestrator_with_agent(
        _MentorToolkit(),
        agent,
        realtime_wait=True,
        tool_workers=2,
    )
    orch.initialize()
    orch.step()

    assert second_done.wait(timeout=1)
    assert [job.tool_calls[0].id for job in orch._pending_agent_tool_jobs] == [
        "call_read_1",
        "call_read_2",
    ]

    orch.step()

    assert [job.tool_calls[0].id for job in orch._pending_agent_tool_jobs] == [
        "call_read_1"
    ]
    assert agent.received[-1]["tool_results"].id == "call_read_2"
    first_tool_tick = orch.ticks[1]
    assert [tool_call.id for tool_call in first_tool_tick.agent_tool_calls] == [
        "call_read_2",
        "call_read_1",
    ]
    assert [result.id for result in first_tool_tick.agent_tool_results] == [
        "call_read_2"
    ]

    release_first.set()
    for _ in range(50):
        if all(job.future.done() for job in orch._pending_agent_tool_jobs):
            break
        time.sleep(0.01)

    orch.step()

    assert not orch._pending_agent_tool_jobs
    assert agent.received[-1]["tool_results"].id == "call_read_1"
    assert [tool_call.id for tool_call in first_tool_tick.agent_tool_calls] == [
        "call_read_2",
        "call_read_1",
    ]
    assert [result.id for result in first_tool_tick.agent_tool_results] == [
        "call_read_2",
        "call_read_1",
    ]
    orch._shutdown_agent_tool_executor()


def test_realtime_wait_runs_mixed_read_write_tool_jobs_in_parallel(monkeypatch):
    release = threading.Event()
    both_started = threading.Event()
    started = []
    lock = threading.Lock()

    def _mark_started(name: str) -> None:
        with lock:
            started.append(name)
            if len(started) == 2:
                both_started.set()

    def _delayed_execute(self, *, tool_calls, **_kwargs):
        tool_call = tool_calls[0]
        _mark_started(f"read:{tool_call.id}")
        release.wait(timeout=2)
        result = ToolMessage(
            id=tool_call.id,
            role="tool",
            requestor="assistant",
            content='{"ok": true}',
            error=False,
        )
        return tool_calls, [result], [result], []

    def _delayed_allow(
        self,
        tool_call,
        recent_transcript,
        recent_tool_history=None,
        history_events=None,
    ):
        _mark_started(f"write:{tool_call.id}")
        release.wait(timeout=2)
        return (
            MentorDecision(
                decision="allow",
                agent_note="After this write, confirm the cancellation.",
            ),
            MentorEvent(
                stage="pre_write",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_type="write",
                mutates_state=True,
                outcome="allow",
                elapsed_ms=0.0,
                decision="allow",
            ),
            None,
        )

    monkeypatch.setattr(
        FullDuplexOrchestrator,
        "_execute_agent_tool_calls_with_mentor_context",
        _delayed_execute,
    )
    monkeypatch.setattr(ToolBoundaryMentor, "pre_write_gate", _delayed_allow)
    toolkit = _MentorToolkit()
    orch = _orchestrator_with_agent(
        toolkit,
        _ScriptedReadWriteAgent(),
        realtime_wait=True,
        tool_workers=2,
    )
    orch.initialize()

    orch.step()

    assert both_started.wait(timeout=1)
    scheduled_events = [
        event
        for event in orch._voice_timing_trace["async_tool_jobs"]
        if event["event"] == "scheduled"
    ]
    assert [event["executor"] for event in scheduled_events] == [
        "parallel",
        "parallel",
    ]
    assert sorted(started) == ["read:call_read", "write:call_write"]

    release.set()
    for _ in range(50):
        if all(job.future.done() for job in orch._pending_agent_tool_jobs):
            break
        time.sleep(0.01)

    orch.step()

    assert toolkit.cancelled is True
    assert not orch._pending_agent_tool_jobs
    completed_events = [
        event
        for event in orch._voice_timing_trace["async_tool_jobs"]
        if event["event"] == "completed"
    ]
    assert [event["executor"] for event in completed_events] == [
        "parallel",
        "parallel",
    ]
    orch._shutdown_agent_tool_executor()


def test_realtime_wait_replay_order_follows_async_write_execution(monkeypatch):
    release_first = threading.Event()
    second_done = threading.Event()

    def _delayed_allow(
        self,
        tool_call,
        recent_transcript,
        recent_tool_history=None,
        history_events=None,
    ):
        if tool_call.id == "call_write_1":
            release_first.wait(timeout=2)
        if tool_call.id == "call_write_2":
            second_done.set()
        return (
            MentorDecision(decision="allow"),
            MentorEvent(
                stage="pre_write",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_type="write",
                mutates_state=True,
                outcome="allow",
                elapsed_ms=0.0,
                decision="allow",
            ),
            None,
        )

    monkeypatch.setattr(ToolBoundaryMentor, "pre_write_gate", _delayed_allow)
    toolkit = _MentorToolkit()
    orch = _orchestrator_with_agent(
        toolkit,
        _ScriptedTwoWriteAgent(),
        realtime_wait=True,
        tool_workers=2,
    )
    orch.initialize()
    orch.step()

    assert second_done.wait(timeout=1)
    orch.step()

    first_tool_tick = orch.ticks[1]
    assert toolkit.write_log == ["second"]
    assert [job.tool_calls[0].id for job in orch._pending_agent_tool_jobs] == [
        "call_write_1"
    ]
    assert [tool_call.id for tool_call in first_tool_tick.agent_tool_calls] == [
        "call_write_2",
        "call_write_1",
    ]
    assert [result.id for result in first_tool_tick.agent_tool_results] == [
        "call_write_2"
    ]

    release_first.set()
    for _ in range(50):
        if all(job.future.done() for job in orch._pending_agent_tool_jobs):
            break
        time.sleep(0.01)

    orch.step()

    assert toolkit.write_log == ["second", "first"]
    assert [tool_call.id for tool_call in first_tool_tick.agent_tool_calls] == [
        "call_write_2",
        "call_write_1",
    ]
    assert [result.id for result in first_tool_tick.agent_tool_results] == [
        "call_write_2",
        "call_write_1",
    ]
    orch._shutdown_agent_tool_executor()


def test_realtime_wait_write_executes_only_after_gate_completion(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def _delayed_allow(
        self,
        tool_call,
        recent_transcript,
        recent_tool_history=None,
        history_events=None,
    ):
        started.set()
        release.wait(timeout=2)
        return (
            MentorDecision(
                decision="allow",
                agent_note="After this write, confirm the cancellation.",
            ),
            MentorEvent(
                stage="pre_write",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_type="write",
                mutates_state=True,
                outcome="allow",
                elapsed_ms=0.0,
                decision="allow",
            ),
            None,
        )

    monkeypatch.setattr(ToolBoundaryMentor, "pre_write_gate", _delayed_allow)
    toolkit = _MentorToolkit()
    agent = _ScriptedWriteAgent()
    orch = _orchestrator_with_agent(
        toolkit,
        agent,
        realtime_wait=True,
    )
    orch.initialize()

    orch.step()

    assert started.wait(timeout=1)
    first_tool_tick = orch.ticks[-1]
    assert [tool_call.id for tool_call in first_tool_tick.agent_tool_calls] == [
        "call_write"
    ]
    assert first_tool_tick.agent_tool_results == []
    assert toolkit.cancelled is False

    release.set()
    for _ in range(20):
        if orch._pending_agent_tool_jobs[0].future.done():
            break
        time.sleep(0.01)

    orch.step()

    assert toolkit.cancelled is True
    assert [result.id for result in first_tool_tick.agent_tool_results] == [
        "call_write"
    ]
    assert agent.received[-1]["tool_results"].id == "call_write"
    official_payload = json.loads(first_tool_tick.agent_tool_results[0].content or "{}")
    provider_payload = json.loads(agent.received[-1]["tool_results"].content or "{}")
    assert "mentor_note" not in official_payload
    assert provider_payload["mentor_note"] == (
        "After this write, confirm the cancellation."
    )
    provider_trace = orch._voice_timing_trace["provider_tool_results"][-1]
    assert provider_trace["kind"] == "mentor_augmented_write_result"
    completed_event = orch._voice_timing_trace["async_tool_jobs"][1]
    assert completed_event["kind"] == "pre_write_gate"
    assert completed_event["approved_write_count"] == 1
    assert completed_event["executed_tick_id"] == completed_event["delivered_tick_id"]
    assert (
        orch._voice_timing_trace["tool_events"][-1]["executed_tick_id"]
        == (completed_event["executed_tick_id"])
    )
    orch._shutdown_agent_tool_executor()


def test_realtime_wait_finalize_expires_pending_write_without_mutation(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def _delayed_allow(
        self,
        tool_call,
        recent_transcript,
        recent_tool_history=None,
        history_events=None,
    ):
        started.set()
        release.wait(timeout=2)
        return (
            MentorDecision(decision="allow"),
            MentorEvent(
                stage="pre_write",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_type="write",
                mutates_state=True,
                outcome="allow",
                elapsed_ms=0.0,
                decision="allow",
            ),
            None,
        )

    monkeypatch.setattr(ToolBoundaryMentor, "pre_write_gate", _delayed_allow)
    toolkit = _MentorToolkit()
    orch = _orchestrator_with_agent(
        toolkit,
        _ScriptedWriteAgent(),
        realtime_wait=True,
    )
    orch.initialize()
    orch.step()

    assert started.wait(timeout=1)
    first_tool_tick = orch.ticks[-1]
    assert [tool_call.id for tool_call in first_tool_tick.agent_tool_calls] == [
        "call_write"
    ]

    orch._expire_pending_agent_tool_jobs_for_finalize()

    assert toolkit.cancelled is False
    assert first_tool_tick.agent_tool_calls == []
    assert first_tool_tick.agent_tool_results == []
    expired_event = orch._voice_timing_trace["async_tool_jobs"][-1]
    assert expired_event["event"] == "expired_after_conversation_end"
    assert expired_event["kind"] == "pre_write_gate"
    assert expired_event["official_result_count"] == 0
    release.set()
    orch._shutdown_agent_tool_executor()


def test_blocked_write_output_closes_provider_tool_call():
    toolkit = _MentorToolkit()
    orch = _orchestrator(toolkit)
    tool_call = ToolCall(
        id="call_write",
        name="cancel_order",
        arguments={"order_id": "order_1"},
    )
    _executed_calls, _official_results, provider_results, synthetic_context = (
        orch._execute_agent_tool_calls_with_mentor([tool_call])
    )
    adapter = _CapturingAdapter()
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="Test policy.",
        adapter=adapter,
        vad_config=object(),
    )
    state = agent.get_init_state()

    agent.get_next_chunk(
        state=state,
        tool_results=provider_results[0],
        synthetic_agent_context=synthetic_context,
    )

    assert toolkit.cancelled is False
    sent_tool_result = adapter.tool_result_batches[0][0]
    assert sent_tool_result[0] == "call_write"
    assert json.loads(sent_tool_result[1])["type"] == "mentor_blocked_tool_call"
    assert sent_tool_result[2] is True
    assert synthetic_context == []
    assert adapter.synthetic_context_batches == [[]]


def test_blocked_write_stays_out_of_official_trajectory():
    toolkit = _MentorToolkit()
    agent = _ScriptedWriteAgent()
    orch = _orchestrator_with_agent(toolkit, agent)

    orch.initialize()
    orch.step()
    first_step_tick = orch.ticks[1]

    assert toolkit.cancelled is False
    assert first_step_tick.agent_tool_calls == []
    assert first_step_tick.agent_tool_results == []
    assert first_step_tick.agent_synthetic_context == []
    assert orch.pending_agent_tool_results is not None
    assert json.loads(orch.pending_agent_tool_results.content or "{}")["type"] == (
        "mentor_blocked_tool_call"
    )
    assert FullDuplexActionEvaluator.extract_tool_calls(orch.ticks) == []
    assert all(message.role != "tool" for message in orch.get_messages())


def test_blocked_write_replays_without_fake_tool_result_mismatch():
    toolkit = _MentorToolkit()
    agent = _ScriptedWriteAgent()
    orch = _orchestrator_with_agent(toolkit, agent)

    orch.initialize()
    orch.step()
    reward_info = FullDuplexEnvironmentEvaluator.calculate_reward(
        environment_constructor=lambda **kwargs: Environment(
            domain_name="retail",
            policy="Ask for explicit confirmation before cancellation.",
            tools=_MentorToolkit(),
            **kwargs,
        ),
        task=Task(
            id="mentor_test",
            user_scenario=UserScenario(instructions="Test user scenario."),
            evaluation_criteria=EvaluationCriteria(
                env_assertions=[
                    EnvAssertion(
                        env_type="assistant",
                        func_name="assert_cancelled",
                        arguments={"expected": False},
                    )
                ],
                reward_basis=[RewardType.ENV_ASSERTION],
            ),
        ),
        full_trajectory=orch.ticks,
    )

    assert toolkit.cancelled is False
    assert reward_info.reward == 1.0
    assert reward_info.env_assertions is not None
    assert reward_info.env_assertions[0].met is True


def test_full_duplex_step_delivers_read_result_and_mentor_note_next_tick():
    toolkit = _MentorToolkit()
    agent = _ScriptedReadAgent()
    orch = _orchestrator_with_agent(toolkit, agent)

    orch.initialize()
    orch.step()
    first_step_tick = orch.ticks[1]

    assert len(first_step_tick.agent_tool_calls) == 1
    assert len(first_step_tick.agent_tool_results) == 1
    assert first_step_tick.agent_synthetic_context == []
    assert orch.pending_agent_tool_results is not None
    pending_payload = json.loads(orch.pending_agent_tool_results.content or "{}")
    assert list(pending_payload.keys())[0] == "mentor_note"
    assert "Available variants: 2" in pending_payload["mentor_note"]
    assert orch.pending_agent_synthetic_context == []

    orch.step()
    delivered = agent.received[-1]

    assert delivered["tool_results"] is not None
    delivered_payload = json.loads(delivered["tool_results"].content or "{}")
    assert "Available variants: 2" in delivered_payload["mentor_note"]
    assert delivered["synthetic_agent_context"] is None
    assert orch.pending_agent_tool_results is None
    assert orch.pending_agent_synthetic_context == []
    assert orch.ticks[2].agent_synthetic_context == []
    assert orch._voice_timing_trace["ticks"][0]["agent_synthetic_context_count"] == 0


def test_mentor_model_timeout_returns_without_waiting_for_slow_completion(
    monkeypatch,
):
    def _slow_completion(**kwargs):
        time.sleep(0.2)
        return None

    monkeypatch.setattr(mentor_module, "completion", _slow_completion)
    env = Environment(
        domain_name="retail",
        policy="Test policy.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="xai",
            tool_mentor_enabled=True,
            tool_mentor_read_timeout_seconds=0.01,
        ),
        environment=env,
    )

    start = time.perf_counter()
    try:
        mentor._call_json_model(
            messages=[{"role": "user", "content": "{}"}],
            timeout_seconds=0.01,
            response_format=mentor_module.POST_READ_MENTOR_RESPONSE_FORMAT,
        )
    except FutureTimeout:
        pass
    else:
        raise AssertionError("Expected mentor model timeout")

    assert time.perf_counter() - start < 0.1


def test_mentor_model_receives_reasoning_effort(monkeypatch):
    calls = []

    def _capture_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"agent_note":"Use exact IDs."}')
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Test policy.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
            tool_mentor_reasoning_effort="none",
        ),
        environment=env,
    )

    result = mentor._call_json_model(
        messages=[{"role": "user", "content": "{}"}],
        timeout_seconds=0.1,
        response_format=mentor_module.POST_READ_MENTOR_RESPONSE_FORMAT,
    )

    assert result == {"agent_note": "Use exact IDs."}
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["reasoning_effort"] == "none"
    assert calls[0]["max_tokens"] == mentor_module.MENTOR_JSON_MAX_TOKENS
    response_format = calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "tool_mentor_read_note"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["required"] == ["agent_note"]


def test_mentor_model_writes_debug_log(monkeypatch):
    calls = []
    logs = []

    def _capture_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"agent_note":"Use exact IDs."}')
                )
            ]
        )

    def _capture_log(request_data, response_data, call_name=None):
        logs.append(
            {
                "request": request_data,
                "response": response_data,
                "call_name": call_name,
            }
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    monkeypatch.setattr(mentor_module, "_write_llm_log", _capture_log)
    env = Environment(
        domain_name="retail",
        policy="Test policy.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
        ),
        environment=env,
    )

    mentor._call_json_model(
        messages=[
            {
                "role": "user",
                "content": (
                    '<history_block>\n{"conversation_history":[],"focus":{}}\n'
                    "</history_block>"
                ),
            }
        ],
        timeout_seconds=0.1,
        response_format=mentor_module.POST_READ_MENTOR_RESPONSE_FORMAT,
    )

    assert calls
    assert logs[0]["call_name"] == "tool_mentor_read_note"
    assert logs[0]["request"]["model"] == "gpt-5.5"
    assert logs[0]["request"]["max_tokens"] == mentor_module.MENTOR_JSON_MAX_TOKENS
    assert "<history_block>" in logs[0]["request"]["messages"][0]["content"]
    assert logs[0]["response"]["content"] == '{"agent_note":"Use exact IDs."}'


def test_mentor_thread_preserves_llm_log_context(monkeypatch, tmp_path):
    def _capture_completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"agent_note":"Use exact IDs."}')
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Test policy.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
        ),
        environment=env,
    )

    set_llm_log_dir(tmp_path)
    set_llm_log_mode("all")
    try:
        mentor._call_json_model(
            messages=[
                {
                    "role": "user",
                    "content": (
                        '<history_block>\n{"conversation_history":[],"focus":{}}\n'
                        "</history_block>"
                    ),
                }
            ],
            timeout_seconds=0.1,
            response_format=mentor_module.POST_READ_MENTOR_RESPONSE_FORMAT,
        )
    finally:
        set_llm_log_dir(None)
        set_llm_log_mode("latest")

    log_files = list(tmp_path.glob("*tool_mentor_read_note*.json"))
    assert len(log_files) == 1
    payload = json.loads(log_files[0].read_text())
    assert payload["call_name"] == "tool_mentor_read_note"
    assert "<history_block>" in payload["request"]["messages"][0]["content"]


def test_mentor_uses_captured_llm_log_dir_when_call_context_is_empty(
    monkeypatch, tmp_path
):
    def _capture_completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"agent_note":"Use exact IDs."}')
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Test policy.",
        tools=_MentorToolkit(),
    )
    set_llm_log_dir(tmp_path)
    set_llm_log_mode("all")
    try:
        mentor = ToolBoundaryMentor(
            config=AudioNativeConfig(
                provider="openai",
                tool_mentor_enabled=True,
                tool_mentor_model="gpt-5.5",
            ),
            environment=env,
        )
        set_llm_log_dir(None)
        mentor._call_json_model(
            messages=[
                {
                    "role": "user",
                    "content": (
                        '<history_block>\n{"conversation_history":[],"focus":{}}\n'
                        "</history_block>"
                    ),
                }
            ],
            timeout_seconds=0.1,
            response_format=mentor_module.POST_READ_MENTOR_RESPONSE_FORMAT,
        )
    finally:
        set_llm_log_dir(None)
        set_llm_log_mode("latest")

    log_files = list(tmp_path.glob("*tool_mentor_read_note*.json"))
    assert len(log_files) == 1
    payload = json.loads(log_files[0].read_text())
    assert payload["call_name"] == "tool_mentor_read_note"
    assert payload["request"]["model"] == "gpt-5.5"
    assert payload["response"]["content"] == '{"agent_note":"Use exact IDs."}'


def test_mentor_pre_write_uses_structured_output_schema(monkeypatch):
    calls = []

    def _capture_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"decision":"block","agent_note":"Ask for confirmation.",'
                            '"missing_requirements":["explicit_confirmation"]}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Ask for explicit confirmation before cancellation. " + ("x" * 5000),
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
            tool_mentor_reasoning_effort="none",
        ),
        environment=env,
    )

    decision = mentor._llm_pre_write(
        ToolCall(
            id="call_write",
            name="cancel_order",
            arguments={"order_id": "order_1"},
        ),
        recent_transcript=[],
    )

    assert decision.decision == "block"
    assert decision.missing_requirements == ["explicit_confirmation"]
    response_format = calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "tool_mentor_write_decision"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["decision"]["enum"] == ["allow", "block"]
    system_prompt = calls[0]["messages"][0]["content"]
    # Stage discriminator and packet contract only; prompt wording is
    # intentionally unpinned (human litmus review owns content).
    assert "proposed action in <current_tool_event>" in system_prompt
    assert "history_block" in system_prompt
    assert "voice agent policy is below" in system_prompt
    assert "x" * 5000 in system_prompt
    assert "target pre_write" not in system_prompt
    assert "target post_read" not in system_prompt


def test_mentor_post_read_prompt_guides_contextual_notes(monkeypatch):
    calls = []

    def _capture_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"agent_note":"Use exact IDs."}')
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Ask for explicit confirmation before cancellation.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
        ),
        environment=env,
    )

    note = mentor._llm_post_read(
        ToolCall(
            id="call_read",
            name="get_order_details",
            arguments={"order_id": "order_1"},
        ),
        ToolMessage(
            id="call_read",
            role="tool",
            content='{"order_id":"order_1","status":"pending"}',
        ),
        recent_transcript=[],
    )

    assert note == "Use exact IDs."
    system_prompt = calls[0]["messages"][0]["content"]
    # Stage discriminator only; prompt wording is intentionally unpinned
    # (human litmus review owns content).
    assert "Use this tool result to help the voice agent act" in system_prompt


def test_tool_mentor_deadline_defaults_are_seven_seconds():
    config = AudioNativeConfig(provider="openai")

    assert config.tool_mentor_read_timeout_seconds == 7.0
    assert config.tool_mentor_write_timeout_seconds == 7.0
    assert config.tool_mentor_realtime_workers == 5
    alias_config = AudioNativeConfig(
        provider="openai",
        tool_mentor_realtime_read_workers=2,
    )
    assert alias_config.tool_mentor_realtime_workers == 2
    assert "tool_mentor_realtime_read_workers" not in alias_config.model_dump()


def test_mentor_pre_write_packet_includes_recent_tool_history(monkeypatch):
    calls = []

    def _capture_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"decision":"allow","agent_note":"State is verified.",'
                            '"missing_requirements":[]}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Ask for explicit confirmation before cancellation.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
        ),
        environment=env,
    )
    recent_tool_history = [
        {
            "tool_call_id": "call_read",
            "tool_name": "get_order_details",
            "arguments": {"order_id": "order_1"},
            "official_result": '{"order_id":"order_1","status":"pending"}',
            "official_error": False,
        }
    ]

    decision = mentor._llm_pre_write(
        ToolCall(
            id="call_write",
            name="cancel_order",
            arguments={"order_id": "order_1"},
        ),
        recent_transcript=[],
        recent_tool_history=recent_tool_history,
    )

    assert decision.decision == "allow"
    payload = _history_payload_from_message(calls[0]["messages"][1])
    current_tool_event = _current_tool_event_from_message(calls[0]["messages"][1])
    assert payload["history_block_version"] == "agent_observed_history_v3"
    assert "focus" not in payload
    assert current_tool_event["type"] == "proposed_write_tool_call"
    assert current_tool_event["tool_call"]["tool_name"] == "cancel_order"
    tool_result_events = [
        item
        for item in payload["conversation_history"]
        if item["kind"] == "voice_agent_visible_tool_result"
    ]
    assert tool_result_events[0]["content"] == recent_tool_history[0]["official_result"]


def test_mentor_structured_output_validation_rejects_extra_keys(monkeypatch):
    def _capture_completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"agent_note":"Use exact IDs.","extra":"bad"}'
                    )
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Test policy.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
        ),
        environment=env,
    )

    context, event = mentor.post_read_note(
        tool_call=ToolCall(
            id="call_read",
            name="get_product_details",
            arguments={"product_id": "p1"},
        ),
        tool_result=ToolMessage(
            id="call_read",
            role="tool",
            content='{"product_id":"p1"}',
        ),
        recent_transcript=[],
    )

    assert context is None
    assert event.outcome == "error"
    assert "extra keys" in (event.error or "")


def test_mentor_structured_output_validation_rejects_overlong_notes(monkeypatch):
    def _capture_completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"agent_note": "x" * 241})
                    )
                )
            ]
        )

    monkeypatch.setattr(mentor_module, "completion", _capture_completion)
    env = Environment(
        domain_name="retail",
        policy="Test policy.",
        tools=_MentorToolkit(),
    )
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="openai",
            tool_mentor_enabled=True,
            tool_mentor_model="gpt-5.5",
        ),
        environment=env,
    )

    context, event = mentor.post_read_note(
        tool_call=ToolCall(
            id="call_read",
            name="get_product_details",
            arguments={"product_id": "p1"},
        ),
        tool_result=ToolMessage(
            id="call_read",
            role="tool",
            content='{"product_id":"p1"}',
        ),
        recent_transcript=[],
    )

    assert context is None
    assert event.outcome == "error"
    assert "maxLength" in (event.error or "")


def test_finalize_drain_attaches_completed_terminal_tool_job(monkeypatch):
    def _instant_execute(self, *, tool_calls, **_kwargs):
        result = ToolMessage(
            id=tool_calls[0].id,
            role="tool",
            requestor="assistant",
            content='{"ok": true}',
            error=False,
        )
        return tool_calls, [result], [result], []

    monkeypatch.setattr(
        FullDuplexOrchestrator,
        "_execute_agent_tool_calls_with_mentor_context",
        _instant_execute,
    )
    agent = _ScriptedReadAgent()
    orch = _orchestrator_with_agent(_MentorToolkit(), agent, realtime_wait=True)
    orch.initialize()
    orch.step()
    tool_tick = orch.ticks[-1]
    for _ in range(200):
        if (
            orch._pending_agent_tool_jobs
            and orch._pending_agent_tool_jobs[0].future.done()
        ):
            break
        time.sleep(0.01)
    assert orch._pending_agent_tool_jobs
    assert tool_tick.agent_tool_results == []

    orch._drain_pending_agent_tool_jobs_for_finalize()

    assert not orch._pending_agent_tool_jobs
    assert [result.id for result in tool_tick.agent_tool_results] == ["call_read"]
    async_events = orch._voice_timing_trace["async_tool_jobs"]
    assert [event["event"] for event in async_events] == ["scheduled", "completed"]
    orch._shutdown_agent_tool_executor()


def test_finalize_drain_waits_for_in_flight_terminal_tool_job(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def _delayed_execute(self, *, tool_calls, **_kwargs):
        started.set()
        release.wait(timeout=5)
        result = ToolMessage(
            id=tool_calls[0].id,
            role="tool",
            requestor="assistant",
            content='{"ok": true}',
            error=False,
        )
        return tool_calls, [result], [result], []

    monkeypatch.setattr(
        FullDuplexOrchestrator,
        "_execute_agent_tool_calls_with_mentor_context",
        _delayed_execute,
    )
    agent = _ScriptedReadAgent()
    orch = _orchestrator_with_agent(_MentorToolkit(), agent, realtime_wait=True)
    orch.initialize()
    orch.step()
    tool_tick = orch.ticks[-1]
    assert started.wait(timeout=1)
    assert orch._pending_agent_tool_jobs
    assert not orch._pending_agent_tool_jobs[0].future.done()

    timer = threading.Timer(0.2, release.set)
    timer.start()
    orch._drain_pending_agent_tool_jobs_for_finalize()
    timer.cancel()

    assert not orch._pending_agent_tool_jobs
    assert [result.id for result in tool_tick.agent_tool_results] == ["call_read"]
    orch._shutdown_agent_tool_executor()


# ---------------------------------------------------------------------------
# Stop interception: a stop stamped by a pre-gated tool call
# defers termination to the gate outcome, so a blocked hand-off behaves like
# a blocked write instead of silently ending the conversation.
# ---------------------------------------------------------------------------


class _StopToolkit(_MentorToolkit):
    @is_tool(ToolType.GENERIC, mutates_state=False)
    def transfer_to_human_agents(self, summary: str = ""):
        return "Transfer successful"


class _ScriptedTransferAgent:
    STOP_TOKEN = "###STOP###"
    STOP_FUNCTION_NAME = "transfer_to_human_agents"

    def get_next_chunk(
        self,
        state,
        participant_chunk=None,
        tool_results=None,
        synthetic_agent_context=None,
    ):
        self.received.append({"tool_results": tool_results})
        step = state.get("step", 0)
        state["step"] = step + 1
        if step == 0:
            return (
                AssistantMessage(
                    role="assistant",
                    content="Transferring you now. " + self.STOP_TOKEN,
                    contains_speech=False,
                    tool_calls=[
                        ToolCall(
                            id="call_transfer",
                            name=self.STOP_FUNCTION_NAME,
                            arguments={"summary": "customer issue"},
                        )
                    ],
                ),
                state,
            )
        return (
            AssistantMessage(role="assistant", content=None, contains_speech=False),
            state,
        )

    def __init__(self):
        self.received = []

    def get_init_state(self, message_history=None):
        return {"step": 0}

    def set_seed(self, seed: int) -> None:
        return None

    @classmethod
    def is_stop(cls, message) -> bool:
        return message.content is not None and cls.STOP_TOKEN in message.content

    def stop(self, participant_chunk=None, state=None, tool_results=None) -> None:
        return None


class _StubGateDecision:
    def __init__(self, decision: str, agent_note: str = ""):
        self.decision = decision
        self.agent_note = agent_note


class _StubGateEvent:
    def to_trace_dict(self):
        return {"event_type": "pre_write_gate", "stub": True}


def _stop_orchestrator(
    agent,
    *,
    realtime_wait: bool = True,
    stop_interception: bool = True,
    escalation_tools=None,
) -> FullDuplexOrchestrator:
    env = Environment(
        domain_name="retail",
        policy="Try available remedies before transferring.",
        tools=_StopToolkit(),
    )
    task = Task(
        id="stop_interception_test",
        user_scenario=UserScenario(instructions="Test user scenario."),
    )
    kwargs = dict(
        provider="xai",
        tool_mentor_enabled=True,
        tool_mentor_mode="heuristic",
        tool_mentor_realtime_wait=realtime_wait,
        tool_mentor_realtime_workers=2,
        tool_mentor_stop_interception=stop_interception,
    )
    if escalation_tools is not None:
        kwargs["tool_mentor_escalation_tools"] = escalation_tools
    config = AudioNativeConfig(**kwargs)
    return FullDuplexOrchestrator(
        domain="retail",
        agent=agent,
        user=_FakeUser(),
        environment=env,
        task=task,
        tool_mentor_config=config,
    )


def _mock_gate(orch, decision: str, note: str = ""):
    def _gate(**_kwargs):
        return _StubGateDecision(decision, note), _StubGateEvent(), None

    orch.tool_mentor.pre_write_gate = _gate


def _wait_for_pending_jobs(orch, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        jobs = orch._pending_agent_tool_jobs
        if jobs and all(job.future.done() for job in jobs):
            return
        time.sleep(0.01)
    raise AssertionError("pending mentor jobs did not finish in time")


def test_stop_interception_blocked_transfer_continues_conversation():
    agent = _ScriptedTransferAgent()
    orch = _stop_orchestrator(agent, realtime_wait=True)
    _mock_gate(orch, "block", "Name the untried remedy first.")
    orch.initialize()
    orch.step()

    assert orch.done is False
    assert orch.termination_reason is None
    recorded = orch.ticks[-1].agent_chunk
    assert recorded is not None
    assert "###STOP###" not in (recorded.content or "")
    interceptions = orch._voice_timing_trace.get("stop_interceptions")
    assert interceptions and interceptions[0]["event"] == "stop_interception_deferred"
    assert interceptions[0]["tool_names"] == ["transfer_to_human_agents"]

    _wait_for_pending_jobs(orch)
    orch.step()
    orch.step()

    assert orch.done is False
    assert orch.termination_reason is None
    delivered = [
        str(record["tool_results"])
        for record in agent.received
        if record["tool_results"]
    ]
    assert any("mentor_blocked_tool_call" in text for text in delivered)
    orch._shutdown_agent_tool_executor()


def test_stop_interception_allowed_transfer_executes_and_terminates():
    agent = _ScriptedTransferAgent()
    orch = _stop_orchestrator(agent, realtime_wait=True)
    _mock_gate(orch, "allow")
    orch.initialize()
    orch.step()

    assert orch.done is False
    _wait_for_pending_jobs(orch)
    orch.step()

    assert orch.done is True
    assert orch.termination_reason == TerminationReason.AGENT_STOP
    executed = [
        tool_call.name for tick in orch.ticks for tool_call in tick.agent_tool_calls
    ]
    assert "transfer_to_human_agents" in executed
    orch._shutdown_agent_tool_executor()


def test_stop_interception_disabled_keeps_stop_at_proposal():
    agent = _ScriptedTransferAgent()
    orch = _stop_orchestrator(agent, realtime_wait=True, stop_interception=False)
    _mock_gate(orch, "block")
    orch.initialize()
    orch.step()

    assert orch.done is True
    assert orch.termination_reason == TerminationReason.AGENT_STOP
    assert "###STOP###" in (orch.ticks[-1].agent_chunk.content or "")
    orch._shutdown_agent_tool_executor()


def test_stop_interception_inactive_when_stop_tool_not_gated():
    agent = _ScriptedTransferAgent()
    orch = _stop_orchestrator(agent, realtime_wait=True, escalation_tools=[])
    orch.initialize()
    orch.step()

    assert orch.done is True
    assert orch.termination_reason == TerminationReason.AGENT_STOP
    orch._shutdown_agent_tool_executor()


def test_stop_interception_sync_path_allow_and_block():
    agent = _ScriptedTransferAgent()
    orch = _stop_orchestrator(agent, realtime_wait=False)
    _mock_gate(orch, "allow")
    orch.initialize()
    orch.step()
    assert orch.done is True
    assert orch.termination_reason == TerminationReason.AGENT_STOP

    agent2 = _ScriptedTransferAgent()
    orch2 = _stop_orchestrator(agent2, realtime_wait=False)
    _mock_gate(orch2, "block", "Try the remedy first.")
    orch2.initialize()
    orch2.step()
    assert orch2.done is False
    assert orch2.termination_reason is None


def test_terminate_helper_never_overrides_existing_termination():
    agent = _ScriptedTransferAgent()
    orch = _stop_orchestrator(agent)
    orch.done = True
    orch.termination_reason = TerminationReason.USER_STOP
    orch._terminate_if_stop_tool_executed(
        [ToolCall(id="x", name="transfer_to_human_agents", arguments={})]
    )
    assert orch.termination_reason == TerminationReason.USER_STOP
