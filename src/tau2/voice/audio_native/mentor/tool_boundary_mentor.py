"""Agent-side tool boundary mentor for audio-native simulations."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from loguru import logger

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType
from tau2.utils.llm_utils import _write_llm_log, completion, llm_log_dir, llm_log_mode

MentorStage = Literal["pre_write", "post_read"]
# Budget must cover thinking tokens plus the JSON body: gemini-3.x low thinking
# spends ~100+ completion tokens on thoughts before content, so 192 truncated
# mid-JSON (probe 2026-07-02: 6/6 truncation at 192, 0/6 at 1024; usage showed
# reasoning=104 of completion=162). 4096 leaves headroom for longer thinking;
# the model stops on its own and the stage timeout bounds wall time.
MENTOR_JSON_MAX_TOKENS = 4096
MENTOR_VALIDATION_ATTEMPTS = 2
HISTORY_BLOCK_VERSION = "agent_observed_history_v3"

PRE_WRITE_MENTOR_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "tool_mentor_write_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {"type": "string", "enum": ["allow", "block"]},
                "agent_note": {"type": "string", "maxLength": 280},
                "missing_requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["decision", "agent_note", "missing_requirements"],
        },
    },
}

POST_READ_MENTOR_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "tool_mentor_read_note",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "agent_note": {"type": "string", "maxLength": 240},
            },
            "required": ["agent_note"],
        },
    },
}


@dataclass
class SyntheticAgentContext:
    """Synthetic mentor context delivered only to the agent provider."""

    type: str
    synthetic: bool
    parent_tool_call_id: str
    stage: MentorStage
    agent_note: str
    tool_name: str
    decision: Optional[str] = None

    def to_model_context(self) -> str:
        """Render the note for provider-side conversation context."""
        prefix = (
            "SYNTHETIC MENTOR NOTE FOR THE ASSISTANT. "
            "This context is separate from customer speech. "
            "Use it to read tool results and policy."
        )
        return f"{prefix}\nTool: {self.tool_name}\nNote: {self.agent_note}"

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize for trajectory diagnostics."""
        return {
            "type": self.type,
            "synthetic": self.synthetic,
            "parent_tool_call_id": self.parent_tool_call_id,
            "stage": self.stage,
            "agent_note": self.agent_note,
            "tool_name": self.tool_name,
            "decision": self.decision,
        }


@dataclass
class MentorDecision:
    """Pre-write gate decision."""

    decision: Literal["allow", "block", "timeout", "error"]
    agent_note: Optional[str] = None
    missing_requirements: list[str] = field(default_factory=list)
    raw_response: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


@dataclass
class MentorEvent:
    """Telemetry event for one mentor stage."""

    stage: MentorStage
    tool_name: str
    tool_call_id: str
    tool_type: str
    mutates_state: bool
    outcome: str
    elapsed_ms: float
    decision: Optional[str] = None
    agent_note: Optional[str] = None
    error: Optional[str] = None

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize for SimulationRun.info."""
        return {
            "stage": self.stage,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "tool_type": self.tool_type,
            "mutates_state": self.mutates_state,
            "outcome": self.outcome,
            "elapsed_ms": self.elapsed_ms,
            "decision": self.decision,
            "agent_note": self.agent_note,
            "error": self.error,
        }


class ToolBoundaryMentor:
    """LLM-backed mentor for agent-originated tool calls."""

    def __init__(
        self,
        config: AudioNativeConfig,
        environment: Environment,
    ) -> None:
        self.config = config
        self.environment = environment
        self._captured_llm_log_dir = llm_log_dir.get()
        self._captured_llm_log_mode = llm_log_mode.get()
        self._escalation_tools = set(config.tool_mentor_escalation_tools or [])
        self._available_tools_cache: Optional[list[dict[str, str]]] = None

    def get_tool_metadata(self, tool_name: str) -> tuple[ToolType, bool]:
        """Read assistant tool metadata from the environment toolkit."""
        if self.environment.tools is None or not self.environment.tools.has_tool(
            tool_name
        ):
            return ToolType.WRITE, True
        return (
            self.environment.tools.tool_type(tool_name),
            self.environment.tools.tool_mutates_state(tool_name),
        )

    def is_escalation_tool(self, tool_name: str) -> bool:
        """Return whether the tool is configured hand-off/give-up wiring."""
        return tool_name in self._escalation_tools

    def should_pre_gate(self, tool_name: str) -> bool:
        """Return whether the tool should pass through the pre-execution gate.

        Gated: state-mutating writes plus configured escalation (hand-off)
        tools, which end the agent's own chance to fix the problem.
        """
        _tool_type, mutates_state = self.get_tool_metadata(tool_name)
        return mutates_state or self.is_escalation_tool(tool_name)

    def should_post_read_note(self, tool_name: str) -> bool:
        """Return whether the tool result should receive a post-read note."""
        tool_type, mutates_state = self.get_tool_metadata(tool_name)
        return tool_type == ToolType.READ and not mutates_state

    def pre_write_gate(
        self,
        tool_call: ToolCall,
        recent_transcript: list[dict[str, str]],
        recent_tool_history: Optional[list[dict[str, Any]]] = None,
        history_events: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[MentorDecision, MentorEvent, Optional[SyntheticAgentContext]]:
        """Run the pre-write mentor gate."""
        tool_type, mutates_state = self.get_tool_metadata(tool_call.name)
        start = time.perf_counter()
        try:
            if self.config.tool_mentor_mode == "heuristic":
                decision = self._heuristic_pre_write(tool_call, recent_transcript)
            else:
                decision = self._llm_pre_write(
                    tool_call,
                    recent_transcript,
                    recent_tool_history=recent_tool_history,
                    history_events=history_events,
                )
        except FutureTimeout:
            elapsed_ms = (time.perf_counter() - start) * 1000
            decision = MentorDecision(decision="timeout", elapsed_ms=elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"Tool mentor pre-write failed: {exc}")
            decision = MentorDecision(
                decision="error",
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )

        if decision.elapsed_ms == 0.0:
            decision.elapsed_ms = (time.perf_counter() - start) * 1000

        event = MentorEvent(
            stage="pre_write",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            tool_type=tool_type.value,
            mutates_state=mutates_state,
            outcome=decision.decision,
            elapsed_ms=decision.elapsed_ms,
            decision=decision.decision,
            agent_note=decision.agent_note,
            error=decision.error,
        )

        context = None
        if decision.decision == "block" and decision.agent_note:
            context = SyntheticAgentContext(
                type="mentor_control",
                synthetic=True,
                parent_tool_call_id=tool_call.id,
                stage="pre_write",
                agent_note=decision.agent_note,
                tool_name=tool_call.name,
                decision="block",
            )

        return decision, event, context

    def post_read_note(
        self,
        tool_call: ToolCall,
        tool_result: ToolMessage,
        recent_transcript: list[dict[str, str]],
        recent_tool_history: Optional[list[dict[str, Any]]] = None,
        history_events: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[Optional[SyntheticAgentContext], MentorEvent]:
        """Generate a synthetic mentor note after a read tool result."""
        tool_type, mutates_state = self.get_tool_metadata(tool_call.name)
        start = time.perf_counter()
        note: Optional[str] = None
        outcome = "no_note"
        error: Optional[str] = None

        try:
            if self.config.tool_mentor_mode == "heuristic":
                note = self._heuristic_post_read(tool_call, tool_result)
            else:
                note = self._llm_post_read(
                    tool_call,
                    tool_result,
                    recent_transcript,
                    recent_tool_history=recent_tool_history,
                    history_events=history_events,
                )
            outcome = "note" if note else "no_note"
        except FutureTimeout:
            outcome = "timeout"
        except Exception as exc:
            error = str(exc)
            outcome = "error"
            logger.warning(f"Tool mentor post-read failed: {exc}")

        elapsed_ms = (time.perf_counter() - start) * 1000
        event = MentorEvent(
            stage="post_read",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            tool_type=tool_type.value,
            mutates_state=mutates_state,
            outcome=outcome,
            elapsed_ms=elapsed_ms,
            agent_note=note,
            error=error,
        )

        if not note:
            return None, event

        return (
            SyntheticAgentContext(
                type="mentor_note",
                synthetic=True,
                parent_tool_call_id=tool_call.id,
                stage="post_read",
                agent_note=note,
                tool_name=tool_call.name,
            ),
            event,
        )

    @staticmethod
    def augment_tool_result_with_note(
        tool_result: ToolMessage,
        agent_note: str,
    ) -> ToolMessage:
        """Add a provider-visible mentor note while preserving official result traces."""
        content = tool_result.content or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = content

        if isinstance(parsed, dict):
            augmented = {"mentor_note": agent_note}
            augmented.update(parsed)
        else:
            augmented = {
                "mentor_note": agent_note,
                "value": parsed,
            }

        return ToolMessage(
            id=tool_result.id,
            role=tool_result.role,
            content=json.dumps(augmented, ensure_ascii=False),
            requestor=tool_result.requestor,
            error=tool_result.error,
        )

    def _policy_text(self) -> str:
        """Return the full domain policy visible to the voice agent."""
        return self.environment.get_policy()

    def _available_tools_payload(self) -> list[dict[str, str]]:
        """Agent-visible tool inventory so notes reference only real tools."""
        if self._available_tools_cache is None:
            items: list[dict[str, str]] = []
            try:
                tools = self.environment.get_tools()
            except Exception as exc:
                logger.warning(f"Tool mentor could not list agent tools: {exc}")
                tools = []
            for tool in sorted(tools, key=lambda t: t.name):
                description = " ".join((tool.short_desc or "").split())
                items.append({"name": tool.name, "description": description[:140]})
            self._available_tools_cache = items
        return self._available_tools_cache

    def _history_block_messages(
        self,
        *,
        stage: MentorStage,
        tool_call: ToolCall,
        history_events: Optional[list[dict[str, Any]]] = None,
        recent_transcript: Optional[list[dict[str, str]]] = None,
        recent_tool_history: Optional[list[dict[str, Any]]] = None,
        tool_result: Optional[ToolMessage] = None,
    ) -> list[dict[str, str]]:
        events = self._history_events(
            history_events=history_events,
            recent_transcript=recent_transcript or [],
            recent_tool_history=recent_tool_history or [],
        )
        current_tool_event: dict[str, Any]
        if stage == "pre_write":
            event_type = (
                "proposed_handoff_tool_call"
                if self.is_escalation_tool(tool_call.name)
                else "proposed_write_tool_call"
            )
            current_tool_event = {
                "type": event_type,
                "tool_call": self._tool_call_event(tool_call),
            }
        else:
            current_tool_event = {
                "type": "read_tool_result_returned",
                "tool_call": self._tool_call_event(tool_call),
                "official_tool_result": self._official_tool_result_event(tool_result),
            }

        payload = {
            "history_block_version": HISTORY_BLOCK_VERSION,
            "domain": self.environment.get_domain_name(),
            "available_tools": self._available_tools_payload(),
            "conversation_history": events,
        }
        history_block = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        current_block = json.dumps(
            current_tool_event,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return [
            {"role": "system", "content": self._system_prompt(stage)},
            {
                "role": "user",
                "content": (
                    f"<history_block>\n{history_block}\n</history_block>\n"
                    f"<current_tool_event>\n{current_block}\n</current_tool_event>"
                ),
            },
        ]

    def _system_prompt(self, stage: MentorStage) -> str:
        shared = (
            "You are a tool mentor for a real-time customer service voice "
            "agent. The voice agent is optimized for low-latency speech and "
            "can miss details in the customer's request, the policy text, "
            "tool results, or the earlier conversation. Your role is to "
            "correct what the voice agent is misreading or missing, using "
            "the customer's stated request and the policy as the source of "
            "truth, and to briefly state remaining obligations or the next "
            "action. At each tool boundary, identify the one or few details "
            "the voice agent is most likely to miss next. Read the "
            "conversation and the policy to decide which exact values matter "
            "here - identifiers, names, quantities, amounts, dates, "
            "addresses, options, and stated constraints - and preserve those "
            "values exactly as heard or returned. Keep agent_note under 180 "
            "characters, concrete, and avoid restating background. Return an "
            "empty agent_note when there is nothing material to correct or "
            "guide. When correcting a value heard from the customer, mark it "
            "as a guess and propose verification, such as asking the "
            "customer to spell or restate it; state a value as fact only "
            "when it appears in a tool result or in the customer's literal "
            "words. "
            "Tool lookups may compare values exactly as written, so a "
            "correct value in a different formatting can fail. When a "
            "lookup fails on a value the customer already stated, name "
            "the untried common renderings of that value (for example "
            "dash-separated groups, bare digits) and suggest retrying "
            "them before asking the customer again. "
            "You will receive <history_block> with the prior "
            "conversation and an available_tools list naming every tool the "
            "voice agent can call, plus <current_tool_event> with the tool "
            "call or result to mentor. Reference only tools named in "
            "available_tools. The voice agent policy is below.\n\n"
            "<voice_agent_policy>\n"
            f"{self._policy_text()}\n"
            "</voice_agent_policy>\n\n"
        )
        if stage == "pre_write":
            return (
                shared + "Decide whether the proposed action in "
                "<current_tool_event> matches the customer's request and the "
                "policy. Allow it when it is confirmed, policy-valid, and "
                "does not damage remaining obligations. Block it when it "
                "acts on the wrong entity or value, violates a constraint "
                "the customer stated, lacks a confirmation the policy "
                "requires, covers only part of the request in a way that "
                "forecloses the rest, or changes state in a way that can "
                "break remaining work. Hand-off or give-up actions end the "
                "agent's chance to resolve the problem itself: block them "
                "when the policy provides remedies within the agent's own "
                "authority that remain untried, and name those remedies in "
                "the note; allow them when the policy requires human "
                "handling or nothing within the agent's authority is left. "
                "The policy and the available tools outrank the voice "
                "agent's own claims. Remedies include steps the agent "
                "performs by instructing the customer on their device, such "
                "as running checks or changing settings. Before allowing a "
                "hand-off or give-up action, enumerate the policy steps for "
                "the customer's reported issue that do not yet appear in the "
                "conversation record; if any remain, block and name them as "
                "the next steps. For block, state the problem, the next "
                "exact action or question, and the condition under which "
                "proposing this action again would be appropriate. For "
                "allow, include a short note only when important obligations "
                "remain after this action."
            )
        return (
            shared + "Use this tool result to help the voice agent act on "
            "the customer's current request. Briefly state facts the agent "
            "may misread, remaining obligations, the next action, or "
            "information the customer still needs to be told. Return an "
            "empty agent_note when the result adds no material correction "
            "or next-step guidance."
        )

    def _history_events(
        self,
        *,
        history_events: Optional[list[dict[str, Any]]],
        recent_transcript: list[dict[str, str]],
        recent_tool_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if history_events is not None:
            events = [dict(item) for item in history_events]
        else:
            events = self._legacy_history_events(
                recent_transcript=recent_transcript,
                recent_tool_history=recent_tool_history,
            )
        return self._renumber_history_events(events)

    def _legacy_history_events(
        self,
        *,
        recent_transcript: list[dict[str, str]],
        recent_tool_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for item in recent_transcript:
            role = item.get("role", "")
            text = item.get("text", "")
            if role == "user":
                events.append(
                    {
                        "kind": "customer_speech_transcript",
                        "source": "agent_observed_stt",
                        "text": text,
                    }
                )
            elif role == "assistant":
                events.append({"kind": "voice_agent_speech", "text": text})
        for item in recent_tool_history:
            call = ToolCall(
                id=str(item.get("tool_call_id") or ""),
                name=str(item.get("tool_name") or ""),
                arguments=dict(item.get("arguments") or {}),
            )
            events.append(self._tool_call_event(call))
            result_content = item.get("provider_result") or item.get("official_result")
            if result_content is not None:
                events.append(
                    {
                        "kind": "voice_agent_visible_tool_result",
                        "tool_call_id": call.id,
                        "content": self._truncate_tool_history_content(
                            str(result_content)
                        ),
                        "error": bool(
                            item.get(
                                "provider_error", item.get("official_error", False)
                            )
                        ),
                        "skipped_by_mentor": bool(item.get("skipped_by_mentor", False)),
                    }
                )
        return events

    @staticmethod
    def _tool_call_event(tool_call: ToolCall) -> dict[str, Any]:
        return {
            "kind": "voice_agent_tool_call",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
        }

    def _official_tool_result_event(
        self,
        tool_result: Optional[ToolMessage],
    ) -> Optional[dict[str, Any]]:
        if tool_result is None:
            return None
        return {
            "kind": "official_tool_result",
            "tool_call_id": tool_result.id,
            "content": self._truncate_tool_history_content(tool_result.content),
            "error": bool(tool_result.error),
        }

    def _truncate_tool_history_content(self, content: Optional[str]) -> str:
        return (content or "")[: self.config.tool_mentor_max_result_chars]

    @staticmethod
    def _renumber_history_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        numbered: list[dict[str, Any]] = []
        for seq, event in enumerate(events, start=1):
            clean = dict(event)
            clean["seq"] = seq
            numbered.append(clean)
        return numbered

    def _call_json_model(
        self,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        def _attempt() -> dict[str, Any]:
            kwargs: dict[str, Any] = {}
            if self.config.tool_mentor_reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.config.tool_mentor_reasoning_effort
            request_timestamp = datetime.now().isoformat()
            start_time = time.perf_counter()
            request_kwargs: dict[str, Any] = {
                "model": self.config.tool_mentor_model,
                "messages": messages,
                "max_tokens": MENTOR_JSON_MAX_TOKENS,
                "response_format": response_format,
                "timeout": timeout_seconds,
                "num_retries": 0,
                **kwargs,
            }
            if not self._uses_provider_default_sampling(self.config.tool_mentor_model):
                request_kwargs["temperature"] = 0
            response = completion(**request_kwargs)
            generation_time_seconds = time.perf_counter() - start_time
            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)
            content = message.content or "{}"
            self._write_mentor_llm_log(
                messages=messages,
                response_format=response_format,
                timeout_seconds=timeout_seconds,
                request_timestamp=request_timestamp,
                response=response,
                content=content,
                refusal=refusal,
                generation_time_seconds=generation_time_seconds,
                extra_kwargs=kwargs,
            )
            if refusal:
                raise ValueError(f"Tool mentor refused structured output: {refusal}")
            data = self._parse_json_object(content)
            try:
                self._validate_structured_response(data, response_format)
            except ValueError as exc:
                raise ValueError(
                    f"{exc} (content_len={len(content)}, "
                    f"content_tail={content[-48:]!r})"
                ) from exc
            return data

        def _call() -> dict[str, Any]:
            last_error: Optional[Exception] = None
            for attempt in range(MENTOR_VALIDATION_ATTEMPTS):
                try:
                    return _attempt()
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    logger.warning(
                        "Tool mentor structured output attempt "
                        f"{attempt + 1}/{MENTOR_VALIDATION_ATTEMPTS} failed: {exc}"
                    )
            assert last_error is not None
            raise last_error

        executor = ThreadPoolExecutor(max_workers=1)
        context = copy_context()
        future = executor.submit(context.run, _call)
        try:
            return future.result(timeout=timeout_seconds)
        finally:
            if not future.done():
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    def _write_mentor_llm_log(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        timeout_seconds: float,
        request_timestamp: str,
        response: Any,
        content: str,
        refusal: Optional[str],
        generation_time_seconds: float,
        extra_kwargs: dict[str, Any],
    ) -> None:
        call_name = response_format["json_schema"]["name"]
        request_data = {
            "timestamp": request_timestamp,
            "model": self.config.tool_mentor_model,
            "messages": messages,
            "max_tokens": MENTOR_JSON_MAX_TOKENS,
            "response_format": response_format,
            "timeout": timeout_seconds,
            "num_retries": 0,
            "kwargs": extra_kwargs,
        }
        if self._uses_provider_default_sampling(self.config.tool_mentor_model):
            request_data["sampling"] = "provider_default"
        else:
            request_data["temperature"] = 0
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "refusal": refusal,
            "generation_time_seconds": generation_time_seconds,
            "raw_response": self._safe_response_dict(response),
        }
        current_llm_log_dir = llm_log_dir.get()
        _write_llm_log(request_data, response_data, call_name=call_name)
        if current_llm_log_dir is not None:
            return
        if self._captured_llm_log_dir is not None:
            self._write_llm_log_to_dir(
                request_data=request_data,
                response_data=response_data,
                call_name=call_name,
                log_mode=self._captured_llm_log_mode,
            )

    def _write_llm_log_to_dir(
        self,
        *,
        request_data: dict[str, Any],
        response_data: dict[str, Any],
        call_name: str,
        log_mode: str,
    ) -> None:
        log_dir = self._captured_llm_log_dir
        if log_dir is None:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_mode == "latest":
            for existing_file in list(log_dir.glob(f"*_{call_name}_*.json")):
                try:
                    existing_file.unlink()
                except FileNotFoundError:
                    pass

        call_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
        call_data = {
            "call_id": call_id,
            "call_name": call_name,
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data,
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(call_data, f, indent=2)

    @staticmethod
    def _safe_response_dict(response: Any) -> Any:
        if hasattr(response, "to_dict"):
            try:
                return response.to_dict()
            except Exception as exc:
                return {"serialization_error": str(exc), "response": str(response)}
        return str(response)

    def _llm_pre_write(
        self,
        tool_call: ToolCall,
        recent_transcript: list[dict[str, str]],
        recent_tool_history: Optional[list[dict[str, Any]]] = None,
        history_events: Optional[list[dict[str, Any]]] = None,
    ) -> MentorDecision:
        messages = self._history_block_messages(
            stage="pre_write",
            tool_call=tool_call,
            history_events=history_events,
            recent_transcript=recent_transcript,
            recent_tool_history=recent_tool_history,
        )
        start = time.perf_counter()
        data = self._call_json_model(
            messages,
            self.config.tool_mentor_write_timeout_seconds,
            PRE_WRITE_MENTOR_RESPONSE_FORMAT,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_decision = data.get("decision", "allow")
        decision = "block" if raw_decision == "block" else "allow"
        return MentorDecision(
            decision=decision,
            agent_note=data.get("agent_note"),
            missing_requirements=list(data.get("missing_requirements") or []),
            raw_response=data,
            elapsed_ms=elapsed_ms,
        )

    def _llm_post_read(
        self,
        tool_call: ToolCall,
        tool_result: ToolMessage,
        recent_transcript: list[dict[str, str]],
        recent_tool_history: Optional[list[dict[str, Any]]] = None,
        history_events: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[str]:
        messages = self._history_block_messages(
            stage="post_read",
            tool_call=tool_call,
            history_events=history_events,
            recent_transcript=recent_transcript,
            recent_tool_history=recent_tool_history,
            tool_result=tool_result,
        )
        data = self._call_json_model(
            messages,
            self.config.tool_mentor_read_timeout_seconds,
            POST_READ_MENTOR_RESPONSE_FORMAT,
        )
        note = data.get("agent_note")
        return str(note) if note else None

    def _heuristic_pre_write(
        self,
        tool_call: ToolCall,
        recent_transcript: list[dict[str, str]],
    ) -> MentorDecision:
        joined = " ".join(
            item.get("text", "")
            for item in recent_transcript
            if item.get("role") == "user"
        ).lower()
        risky = any(
            word in tool_call.name
            for word in ("cancel", "refund", "exchange", "update", "modify")
        )
        confirmed = any(
            phrase in joined
            for phrase in ("confirm", "yes", "go ahead", "please cancel", "do it")
        )
        if risky and not confirmed:
            return MentorDecision(
                decision="block",
                agent_note=(
                    "Ask the customer for explicit confirmation before running "
                    f"{tool_call.name}."
                ),
                missing_requirements=["explicit_confirmation"],
            )
        return MentorDecision(decision="allow")

    def _heuristic_post_read(
        self,
        tool_call: ToolCall,
        tool_result: ToolMessage,
    ) -> Optional[str]:
        content = tool_result.content or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict) and isinstance(parsed.get("variants"), dict):
            variants = parsed["variants"].values()
            available = sum(
                1 for item in variants if self._truthy_available(item.get("available"))
            )
            return (
                "Count only variants with available=true. "
                f"Available variants: {available}."
            )

        if tool_call.name.startswith("get_") or tool_call.name.startswith("list_"):
            return "Use exact IDs, dates, prices, and availability from this result."

        return None

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        """Parse the first JSON object from a model response."""
        content = content.strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end < start:
                return {}
            data = json.loads(content[start : end + 1])
        if isinstance(data, dict):
            return data
        return {}

    @staticmethod
    def _validate_structured_response(
        data: dict[str, Any],
        response_format: dict[str, Any],
    ) -> None:
        """Validate the small mentor schema after API-side structured output."""
        schema = response_format["json_schema"]["schema"]
        properties = schema.get("properties", {})
        allowed = set(properties)
        required = set(schema.get("required", []))
        missing = sorted(key for key in required if key not in data)
        extra = sorted(key for key in data if key not in allowed)
        if missing:
            raise ValueError(f"Tool mentor response missing keys: {missing}")
        if schema.get("additionalProperties") is False and extra:
            raise ValueError(f"Tool mentor response had extra keys: {extra}")

        for key, field_schema in properties.items():
            if key not in data:
                continue
            value = data[key]
            expected_type = field_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                raise ValueError(f"Tool mentor response key {key} must be a string")
            max_length = field_schema.get("maxLength")
            if (
                expected_type == "string"
                and isinstance(value, str)
                and max_length is not None
                and len(value) > max_length
            ):
                raise ValueError(
                    f"Tool mentor response key {key} exceeds maxLength {max_length}"
                )
            if expected_type == "array":
                if not isinstance(value, list):
                    raise ValueError(f"Tool mentor response key {key} must be an array")
                item_schema = field_schema.get("items", {})
                if item_schema.get("type") == "string" and not all(
                    isinstance(item, str) for item in value
                ):
                    raise ValueError(
                        f"Tool mentor response key {key} must contain strings"
                    )
            enum = field_schema.get("enum")
            if enum is not None and value not in enum:
                raise ValueError(
                    f"Tool mentor response key {key} must be one of {enum}"
                )

    @staticmethod
    def _truthy_available(value: Any) -> bool:
        """Interpret availability values from official tool JSON."""
        if value is True:
            return True
        if isinstance(value, str):
            return value.lower() == "true"
        return False

    @staticmethod
    def _uses_provider_default_sampling(model: str) -> bool:
        """Use provider sampling defaults for reasoning models that own sampling."""
        normalized = model.lower()
        providerless = normalized.split("/", 1)[-1]
        return "gemini-3" in providerless or providerless.startswith("gpt-5")
