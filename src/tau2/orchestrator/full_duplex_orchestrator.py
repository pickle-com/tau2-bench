"""
Full-duplex orchestrator for streaming/voice communication.

This extends the BaseOrchestrator to support simultaneous bidirectional
communication between agent and user.
"""

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeVar, Union

from loguru import logger

from tau2.agent.base.streaming import compute_responsiveness_info
from tau2.agent.base_agent import FullDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import (
    AudioNativeConfig,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, BaseOrchestrator
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import FullDuplexUser
from tau2.utils.llm_utils import get_cost
from tau2.utils.utils import get_now
from tau2.voice.audio_native.mentor import (
    SyntheticAgentContext,
    ToolBoundaryMentor,
)
from tau2.voice.utils.transcript_utils import compute_proportional_user_transcripts

# Type variables for generic full-duplex orchestrator
StreamingAgentT = TypeVar("StreamingAgentT", bound=FullDuplexAgent)
StreamingUserT = TypeVar("StreamingUserT", bound=FullDuplexUser)


@dataclass
class _PendingAgentToolJob:
    """Background agent tool execution that should not pause audio ticks."""

    future: Future
    tool_calls: list[ToolCall]
    started_tick_id: int
    started_perf: float
    kind: Literal["execute", "pre_write_gate"]
    executor_name: Literal["parallel"]


@dataclass
class _AgentToolJobResult:
    """Result produced by one background agent tool job."""

    trajectory_tool_calls: list[ToolCall]
    tool_results: list[ToolMessage]
    provider_results: list[ToolMessage]
    synthetic_contexts: list[dict]
    approved_write_calls: list[ToolCall]
    approved_write_notes: dict[str, str]


class FullDuplexOrchestrator(BaseOrchestrator[StreamingAgentT, StreamingUserT, Tick]):
    """
    Orchestrator for full-duplex (streaming) communication.

    Extends the BaseOrchestrator to support:
    - Simultaneous agent and user communication
    - Chunk-based message passing
    - Tick-based trajectory that groups concurrent events

    Key differences from half-duplex (Orchestrator):
    - Uses get_next_chunk() instead of generate_next_message()
    - Both agent and user can "speak" at the same time
    - Messages are chunked rather than complete
    - Trajectory is organized by ticks, not individual messages

    Trajectory Structure:
    - get_trajectory() returns a list of Tick objects
    - Each Tick contains all events from a single simulation step
    - Tool calls are queued and executed within the tick
    - Participants receive incoming chunks per tick
    - Use get_messages() to get a flat message list
    """

    def __init__(
        self,
        domain: str,
        agent: StreamingAgentT,
        user: StreamingUserT,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        simulation_id: Optional[str] = None,
        tick_duration_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
        tool_mentor_config: Optional[AudioNativeConfig] = None,
    ):
        """
        Initialize FullDuplexOrchestrator.

        Args:
            domain: The domain name of the simulation.
            agent: The streaming agent instance (must have get_next_chunk method).
            user: The streaming user instance (must have get_next_chunk method).
            environment: The environment instance.
            task: The task specification.
            max_steps: Maximum number of simulation steps.
            max_errors: Maximum number of tool execution errors.
            seed: Optional random seed for reproducibility.
            simulation_id: Optional simulation ID.
            tick_duration_seconds: Duration of each simulation tick in seconds (for timing metadata).
            timeout: Maximum wallclock time in seconds. None means no timeout.
            tool_mentor_config: Optional audio-native config with tool mentor settings.
        """
        super().__init__(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            simulation_id=simulation_id,
            timeout=timeout,
        )

        # Set mode to FULL_DUPLEX
        self.mode = CommunicationMode.FULL_DUPLEX

        # Full-duplex specific attributes
        self.current_user_chunk: Optional[UserMessage] = None
        self.current_agent_chunk: Optional[AssistantMessage] = None
        self.tick_duration_seconds = tick_duration_seconds

        # Pending tool results: executed this tick, delivered to participant next tick.
        # This decouples tool execution from the audio/tick loop so that:
        # - No silence is injected during tool result delivery
        # - Agent audio from the tool-call tick is preserved
        # - User audio continuity is maintained
        self.pending_agent_tool_results: Optional[Message] = None
        self.pending_user_tool_results: Optional[Message] = None
        self.pending_agent_synthetic_context: list[dict] = []

        self.tool_mentor: Optional[ToolBoundaryMentor] = None
        self.tool_mentor_realtime_wait = False
        self.tool_mentor_stop_interception = False
        if tool_mentor_config is not None and tool_mentor_config.tool_mentor_enabled:
            self.tool_mentor = ToolBoundaryMentor(
                config=tool_mentor_config,
                environment=environment,
            )
            self.tool_mentor_realtime_wait = (
                tool_mentor_config.tool_mentor_realtime_wait
            )
            self.tool_mentor_stop_interception = (
                tool_mentor_config.tool_mentor_stop_interception
            )

        self._agent_tool_executor: Optional[ThreadPoolExecutor] = None
        self._agent_tool_workers = 1
        self._pending_agent_tool_jobs: list[_PendingAgentToolJob] = []
        if self.tool_mentor is not None and self.tool_mentor_realtime_wait:
            self._agent_tool_workers = max(
                1,
                (
                    tool_mentor_config.tool_mentor_realtime_workers
                    if tool_mentor_config is not None
                    else 1
                ),
            )
            self._agent_tool_executor = ThreadPoolExecutor(
                max_workers=self._agent_tool_workers,
                thread_name_prefix="tau2-agent-tool",
            )

        self._voice_timing_trace: dict[str, list[dict[str, Any]]] = {
            "ticks": [],
            "tool_events": [],
            "mentor_events": [],
            "provider_tool_results": [],
            "provider_visible_events": [],
            "synthetic_context": [],
            "async_tool_jobs": [],
        }

        # Tick-based trajectory structure
        self.ticks: list[Tick] = []

        # Validate mode compatibility
        self._validate_mode_compatibility()

    def _validate_mode_compatibility(self):
        """Validate that agent and user support full-duplex streaming."""
        # Check if agent has get_next_chunk method
        if not hasattr(self.agent, "get_next_chunk"):
            raise ValueError(
                f"FULL_DUPLEX mode requires agent to have 'get_next_chunk' method. "
                f"Agent {self.agent.__class__.__name__} does not support streaming."
            )

        # Check if user has get_next_chunk method
        if not hasattr(self.user, "get_next_chunk"):
            raise ValueError(
                f"FULL_DUPLEX mode requires user to have 'get_next_chunk' method. "
                f"User {self.user.__class__.__name__} does not support streaming."
            )

        logger.info(
            f"FullDuplexOrchestrator initialized with "
            f"agent={self.agent.__class__.__name__}, "
            f"user={self.user.__class__.__name__}"
        )

    def initialize(self):
        """Initialize the orchestrator for full-duplex communication."""
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )

        # Full-duplex doesn't support message history initialization yet
        if len(message_history) > 0:
            raise ValueError(
                "Full duplex mode does not yet support message history initialization"
            )

        # Initialize Environment state
        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set seeds
        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        # Full-duplex specific initialization
        self._initialize_full_duplex()

        self.environment.sync_tools()

    def _initialize_full_duplex(self):
        """Initialize full-duplex specific state."""
        # Initialize user and agent states
        self.user_state = self.user.get_init_state()

        # Create first agent message
        # If agent has create_initial_message (audio-native agents), use it to get
        # a message with proper audio content. Otherwise, use the text-only default.
        if hasattr(self.agent, "create_initial_message"):
            first_agent_message = self.agent.create_initial_message()
        else:
            first_agent_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
            first_agent_message.chunk_id = 0
            first_agent_message.is_final_chunk = True
            first_agent_message.timestamp = get_now()

        self.agent_state = self.agent.get_init_state(
            message_history=[first_agent_message]
        )

        # Initialize current message trackers with dummy message
        dummy_agent_message = AssistantMessage(
            role="assistant",
            content=None,
            timestamp=get_now(),
            chunk_id=0,
            is_final_chunk=True,
            contains_speech=False,
        )

        self.current_agent_chunk = first_agent_message

        # Get first user chunk
        self.current_user_chunk, self.user_state = self.user.get_next_chunk(
            self.user_state, participant_chunk=dummy_agent_message
        )

        # Initialize trajectory with first tick
        init_start_time = time.perf_counter()
        first_tick = Tick(
            tick_id=0,
            timestamp=get_now(),
            agent_chunk=deepcopy(self.current_agent_chunk),
            user_chunk=deepcopy(self.current_user_chunk),
            tick_duration_seconds=self.tick_duration_seconds,
        )
        first_tick.wall_clock_duration_seconds = time.perf_counter() - init_start_time
        self.ticks = [first_tick]

    def _format_chunk_summary(self, chunk: Optional[Message], role: str) -> str:
        """Format a chunk for logging with key information."""
        if chunk is None:
            return f"{role}: (none)"

        parts = [f"{role}:"]

        # Content summary
        if chunk.content:
            content_preview = (
                chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
            )
            parts.append(f'  content="{content_preview}"')
        else:
            parts.append("  content=(empty)")

        # Speech indicator
        if hasattr(chunk, "contains_speech"):
            parts.append(f"  speech={chunk.contains_speech}")

        # Chunk metadata
        if hasattr(chunk, "chunk_id") and chunk.chunk_id is not None:
            parts.append(f"  chunk_id={chunk.chunk_id}")
        if hasattr(chunk, "is_final_chunk") and chunk.is_final_chunk is not None:
            parts.append(f"  is_final={chunk.is_final_chunk}")

        # Tool calls
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            tool_names = [tc.name for tc in chunk.tool_calls]
            parts.append(f"  tool_calls={tool_names}")

        return " | ".join(parts)

    def step(self):
        """
        Perform one tick of full-duplex streaming communication.

        Both agent and user generate chunks simultaneously.
        Tool calls are executed within the tick but results are delivered
        on the next tick, keeping audio flow uninterrupted.
        All events are grouped into a single Tick object.
        """
        if self.done:
            raise ValueError("Simulation is done")

        tick_id = len(self.ticks)
        logger.debug(f"Tick {tick_id}. step_count={self.step_count}")

        # Sanity check: stored chunks should never carry tool_calls
        # (tool_calls are stripped and tracked separately)
        if self.current_agent_chunk and self.current_agent_chunk.is_tool_call():
            raise ValueError(
                f"Agent chunk should not have tool_calls at step start: "
                f"{self.current_agent_chunk}"
            )
        if self.current_user_chunk and self.current_user_chunk.is_tool_call():
            raise ValueError(
                f"User chunk should not have tool_calls at step start: "
                f"{self.current_user_chunk}"
            )

        # Create new tick
        tick_start_time = time.perf_counter()
        tick = Tick(
            tick_id=tick_id,
            timestamp=get_now(),
            tick_duration_seconds=self.tick_duration_seconds,
        )

        (
            completed_agent_provider_results,
            completed_agent_synthetic_context,
        ) = self._collect_completed_agent_tool_jobs(tick_id)
        if completed_agent_provider_results:
            self.pending_agent_tool_results = self._wrap_tool_results(
                completed_agent_provider_results
            )
        if completed_agent_synthetic_context:
            self.pending_agent_synthetic_context.extend(
                completed_agent_synthetic_context
            )

        # Both participants receive the other's previous chunk
        incoming_for_user = self.current_agent_chunk
        incoming_for_agent = self.current_user_chunk

        # Log incoming chunks from previous tick
        logger.debug(f"[Tick {tick_id}] Incoming chunks from previous tick:")
        logger.debug(
            f"  → User receives: {self._format_chunk_summary(incoming_for_user, 'agent_chunk')}"
        )
        logger.debug(
            f"  → Agent receives: {self._format_chunk_summary(incoming_for_agent, 'user_chunk')}"
        )

        # --- 1. Process user turn ---
        (
            user_chunk,
            self.user_state,
            user_tool_calls,
            user_tool_results,
            user_pending_tool_results,
            _user_synthetic_context,
        ) = self._process_participant_turn(
            participant=self.user,
            state=self.user_state,
            incoming_chunk=incoming_for_user,
            is_agent=False,
            pending_tool_results=self.pending_user_tool_results,
        )
        tick.user_chunk = deepcopy(user_chunk)
        tick.user_tool_calls = [deepcopy(tc) for tc in user_tool_calls]
        tick.user_tool_results = [deepcopy(r) for r in user_tool_results]
        self.pending_user_tool_results = (
            self._wrap_tool_results(user_pending_tool_results)
            if user_pending_tool_results
            else None
        )

        # --- 2. Process agent turn ---
        (
            agent_chunk,
            self.agent_state,
            agent_tool_calls,
            agent_tool_results,
            agent_pending_tool_results,
            agent_synthetic_context,
        ) = self._process_participant_turn(
            participant=self.agent,
            state=self.agent_state,
            incoming_chunk=incoming_for_agent,
            is_agent=True,
            pending_tool_results=self.pending_agent_tool_results,
            pending_synthetic_context=self.pending_agent_synthetic_context,
        )
        tick.agent_chunk = deepcopy(agent_chunk)
        tick.agent_tool_calls = [deepcopy(tc) for tc in agent_tool_calls]
        tick.agent_tool_results = [deepcopy(r) for r in agent_tool_results]
        if agent_synthetic_context:
            tick.agent_synthetic_context = [
                deepcopy(context) for context in agent_synthetic_context
            ]
        self.pending_agent_tool_results = (
            self._wrap_tool_results(agent_pending_tool_results)
            if agent_pending_tool_results
            else None
        )
        self.pending_agent_synthetic_context = agent_synthetic_context

        # --- 3. Update state and bookkeeping ---
        self.current_user_chunk = user_chunk
        self.current_agent_chunk = agent_chunk

        # Record wall clock duration for this tick
        tick.wall_clock_duration_seconds = time.perf_counter() - tick_start_time
        self._voice_timing_trace["ticks"].append(
            {
                "tick_id": tick_id,
                "orchestrator_tick_wall_ms": tick.wall_clock_duration_seconds * 1000,
                "agent_tool_call_count": len(agent_tool_calls),
                "user_tool_call_count": len(user_tool_calls),
                "agent_synthetic_context_count": len(agent_synthetic_context),
            }
        )

        logger.debug(
            f"[Tick {tick_id}] Wall-clock duration: "
            f"{tick.wall_clock_duration_seconds:.3f}s"
        )
        if (
            self.tick_duration_seconds is not None
            and tick.wall_clock_duration_seconds < self.tick_duration_seconds
        ):
            logger.warning(
                f"[Tick {tick_id}] Completed in "
                f"{tick.wall_clock_duration_seconds:.3f}s, which is less than "
                f"the expected tick duration of {self.tick_duration_seconds:.3f}s. "
                f"Wall-clock pacing may not be enforced by the adapter."
            )

        self.ticks.append(tick)

        self.step_count += 1
        self.environment.sync_tools()

    def _process_participant_turn(
        self,
        participant: Union[StreamingAgentT, StreamingUserT],
        state: Any,
        incoming_chunk: Optional[Message],
        is_agent: bool,
        pending_tool_results: Optional[Message] = None,
        pending_synthetic_context: Optional[list[dict]] = None,
    ) -> tuple[
        Message,
        Any,
        list[ToolCall],
        list[ToolMessage],
        list[ToolMessage],
        list[dict],
    ]:
        """
        Process a participant's turn with dual-channel input.

        The participant receives both channels in a single get_next_chunk call:
        - participant_chunk: speech/audio from the other participant
        - tool_results: results from previously executed tool calls (if any)

        If the participant's response contains tool calls, they are executed
        immediately. The caller is responsible for wrapping them via
        _wrap_tool_results and storing as pending for the next tick.

        Args:
            participant: The agent or user instance.
            state: The current state of the participant.
            incoming_chunk: The incoming message chunk from the other participant.
            is_agent: True if processing agent, False if processing user.
            pending_tool_results: Tool results from previous tick to deliver.
            pending_synthetic_context: Custom mentor notes from previous tick.

        Returns:
            Tuple of (
                chunk,
                new_state,
                trajectory_tool_calls,
                trajectory_tool_results,
                provider_tool_results,
                synthetic_context,
            ).
        """
        participant_name = "AGENT" if is_agent else "USER"

        is_stop = self.agent.is_stop if is_agent else UserSimulator.is_stop
        termination_reason = (
            TerminationReason.AGENT_STOP if is_agent else TerminationReason.USER_STOP
        )

        new_state = state

        if pending_tool_results is not None:
            logger.debug(
                f"  [{participant_name}] Delivering tool results alongside "
                f"participant chunk"
            )

        # Single get_next_chunk call with both channels
        logger.debug(f"  [{participant_name}] Calling get_next_chunk...")
        if is_agent and pending_synthetic_context:
            new_chunk, new_state = participant.get_next_chunk(
                state=new_state,
                participant_chunk=incoming_chunk,
                tool_results=pending_tool_results,
                synthetic_agent_context=pending_synthetic_context,
            )
        else:
            new_chunk, new_state = participant.get_next_chunk(
                state=new_state,
                participant_chunk=incoming_chunk,
                tool_results=pending_tool_results,
            )
        logger.debug(
            f"  [{participant_name}] Returned chunk: "
            f"{self._format_chunk_summary(new_chunk, 'chunk')}"
        )

        # Validate and check stop condition
        if new_chunk.contains_speech:
            new_chunk.validate()
        if is_stop(new_chunk):
            if is_agent and self._should_defer_stop_to_gate(new_chunk):
                logger.info(
                    f"  [{participant_name}] STOP proposal is mentor-gated; "
                    "termination deferred to the gate outcome"
                )
                self._strip_agent_stop_token(new_chunk)
                self._voice_timing_trace.setdefault(
                    "stop_interceptions", []
                ).append(
                    {
                        "event": "stop_interception_deferred",
                        "tick_id": len(self.ticks),
                        "tool_names": [
                            tool_call.name
                            for tool_call in (new_chunk.tool_calls or [])
                        ],
                    }
                )
            else:
                logger.info(
                    f"  [{participant_name}] *** STOP signal detected ***"
                )
                self.done = True
                self.termination_reason = termination_reason

        # Handle tool calls: execute now, deliver results next tick
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolMessage] = []
        provider_tool_results: list[ToolMessage] = []
        synthetic_context: list[dict] = []

        if new_chunk and new_chunk.is_tool_call():
            proposed_tool_calls = list(new_chunk.tool_calls)
            tool_calls = proposed_tool_calls
            tool_names = [tc.name for tc in proposed_tool_calls]
            logger.info(f"  [{participant_name}] Tool calls: {tool_names}")

            if is_agent and self.tool_mentor is not None:
                if self.tool_mentor_realtime_wait:
                    self._schedule_agent_tool_calls_with_mentor(
                        proposed_tool_calls,
                        tick_id=len(self.ticks),
                    )
                    tool_calls = proposed_tool_calls
                    logger.info(
                        f"  [{participant_name}] Scheduled "
                        f"{len(proposed_tool_calls)} tool(s); results pending "
                        f"until mentor/tool job completes"
                    )
                else:
                    (
                        tool_calls,
                        tool_results,
                        provider_tool_results,
                        synthetic_context,
                    ) = self._execute_agent_tool_calls_with_mentor(proposed_tool_calls)
            else:
                tool_results = self._execute_tool_calls(tool_calls)
                provider_tool_results = list(tool_results)

            for tc, result in zip(proposed_tool_calls, provider_tool_results):
                result_preview = (
                    result.content[:100] + "..."
                    if len(result.content) > 100
                    else result.content
                )
                logger.debug(f"  [{participant_name}]   {tc.name}() → {result_preview}")

            # Return a copy with tool_calls stripped so the chunk only carries
            # speech/audio for the other participant. The original chunk (stored
            # in the participant's tick history via record_tick) must keep its
            # tool_calls intact for linearization / tool-result pairing.
            new_chunk = deepcopy(new_chunk)
            new_chunk.tool_calls = None

            logger.info(
                f"  [{participant_name}] Executed {len(tool_results)} tool(s), "
                f"results pending for next tick"
            )

        return (
            new_chunk,
            new_state,
            tool_calls,
            tool_results,
            provider_tool_results,
            synthetic_context,
        )

    def _should_defer_stop_to_gate(self, chunk) -> bool:
        """Defer a stop stamped by a pre-gated tool call to the gate outcome.

        A blocked stop proposal then behaves exactly like a blocked write:
        the mentor note is delivered next tick and the conversation
        continues. An approved proposal executes and terminates as usual.
        """
        if self.tool_mentor is None or not self.tool_mentor_stop_interception:
            return False
        stop_function = getattr(type(self.agent), "STOP_FUNCTION_NAME", None)
        if not stop_function or not chunk.tool_calls:
            return False
        return any(
            tool_call.name == stop_function
            and self.tool_mentor.should_pre_gate(tool_call.name)
            for tool_call in chunk.tool_calls
        )

    def _strip_agent_stop_token(self, chunk) -> None:
        stop_token = getattr(type(self.agent), "STOP_TOKEN", None)
        if not stop_token or chunk.content is None:
            return
        content = chunk.content.replace(" " + stop_token, "").replace(
            stop_token, ""
        )
        chunk.content = content or None

    def _terminate_if_stop_tool_executed(
        self, executed_tool_calls: list[ToolCall]
    ) -> None:
        """Terminate once a gated stop tool has actually executed."""
        if self.done or not self.tool_mentor_stop_interception:
            return
        stop_function = getattr(type(self.agent), "STOP_FUNCTION_NAME", None)
        if not stop_function:
            return
        if any(
            tool_call.name == stop_function for tool_call in executed_tool_calls
        ):
            logger.info(
                "  [AGENT] Gated stop tool executed; ending the conversation"
            )
            self.done = True
            self.termination_reason = TerminationReason.AGENT_STOP

    def _execute_agent_tool_calls_with_mentor(
        self,
        tool_calls: list[ToolCall],
    ) -> tuple[list[ToolCall], list[ToolMessage], list[ToolMessage], list[dict]]:
        """Execute agent tool calls with custom mentor hooks."""
        if self.tool_mentor is None:
            tool_results = self._execute_tool_calls(tool_calls)
            return tool_calls, tool_results, list(tool_results), []

        return self._execute_agent_tool_calls_with_mentor_context(
            tool_calls=tool_calls,
            recent_transcript=self._recent_transcript(),
            recent_tool_history=self._recent_agent_tool_history(),
            history_events=self._recent_agent_observed_history_events(),
        )

    def _execute_agent_tool_calls_with_mentor_context(
        self,
        *,
        tool_calls: list[ToolCall],
        recent_transcript: list[dict[str, str]],
        recent_tool_history: list[dict[str, Any]],
        history_events: list[dict[str, Any]],
    ) -> tuple[list[ToolCall], list[ToolMessage], list[ToolMessage], list[dict]]:
        """Execute mentored agent tool calls using a captured history snapshot."""
        if self.tool_mentor is None:
            tool_results = self._execute_tool_calls(tool_calls)
            return tool_calls, tool_results, list(tool_results), []

        trajectory_tool_calls: list[ToolCall] = []
        tool_results: list[ToolMessage] = []
        provider_tool_results: list[ToolMessage] = []
        synthetic_contexts: list[dict] = []
        mentor_recent_tool_history = deepcopy(recent_tool_history)
        mentor_history_events = deepcopy(history_events)

        for tool_call in tool_calls:
            should_execute = True
            block_note = "Tool call blocked by mentor."
            approved_write_note: Optional[str] = None
            if self.tool_mentor.should_pre_gate(tool_call.name):
                decision, event, _context = self.tool_mentor.pre_write_gate(
                    tool_call=tool_call,
                    recent_transcript=recent_transcript,
                    recent_tool_history=mentor_recent_tool_history,
                    history_events=mentor_history_events,
                )
                self._voice_timing_trace["mentor_events"].append(event.to_trace_dict())

                if decision.decision == "block":
                    should_execute = False
                    if decision.agent_note:
                        block_note = decision.agent_note
                elif decision.agent_note:
                    approved_write_note = decision.agent_note

            if not should_execute:
                provider_tool_results.append(
                    self._make_mentor_blocked_tool_result(
                        tool_call=tool_call,
                        block_note=block_note,
                    )
                )
                self._record_provider_visible_tool_result(
                    tool_call=tool_call,
                    kind="mentor_blocked_tool_call",
                    official_result_present=False,
                    provider_result=provider_tool_results[-1],
                )
                self._voice_timing_trace["tool_events"].append(
                    {
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "requestor": tool_call.requestor,
                        "tool_call_to_tool_result_ms": 0.0,
                        "error": False,
                        "skipped_by_mentor": True,
                    }
                )
                recent_tool_history.append(
                    self._tool_history_entry(
                        tool_call=tool_call,
                        official_result=None,
                        provider_result=provider_tool_results[-1],
                        skipped_by_mentor=True,
                    )
                )
                history_events = self._append_agent_observed_tool_events(
                    history_events,
                    tool_call=tool_call,
                    tool_result=provider_tool_results[-1],
                    skipped_by_mentor=True,
                )
                continue

            tool_start = time.perf_counter()
            tool_result = self.environment.get_response(tool_call)
            tool_elapsed_ms = (time.perf_counter() - tool_start) * 1000
            if tool_result.error:
                self.num_errors += 1
            trajectory_tool_calls.append(tool_call)
            tool_results.append(tool_result)
            self._terminate_if_stop_tool_executed([tool_call])
            self._voice_timing_trace["tool_events"].append(
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "requestor": tool_call.requestor,
                    "tool_call_to_tool_result_ms": tool_elapsed_ms,
                    "error": tool_result.error,
                }
            )

            provider_tool_result = tool_result
            if approved_write_note:
                provider_tool_result = self.tool_mentor.augment_tool_result_with_note(
                    tool_result=tool_result,
                    agent_note=approved_write_note,
                )
                self._record_provider_visible_tool_result(
                    tool_call=tool_call,
                    kind="mentor_augmented_write_result",
                    official_result_present=True,
                    provider_result=provider_tool_result,
                )
            elif self.tool_mentor.should_post_read_note(tool_call.name):
                context, event = self.tool_mentor.post_read_note(
                    tool_call=tool_call,
                    tool_result=tool_result,
                    recent_transcript=recent_transcript,
                    recent_tool_history=mentor_recent_tool_history,
                    history_events=mentor_history_events,
                )
                self._voice_timing_trace["mentor_events"].append(event.to_trace_dict())
                if context is not None:
                    context_dict = self._serialize_synthetic_context(context)
                    self._voice_timing_trace["synthetic_context"].append(context_dict)
                    provider_tool_result = (
                        self.tool_mentor.augment_tool_result_with_note(
                            tool_result=tool_result,
                            agent_note=context.agent_note,
                        )
                    )
                    self._record_provider_visible_tool_result(
                        tool_call=tool_call,
                        kind="mentor_augmented_read_result",
                        official_result_present=True,
                        provider_result=provider_tool_result,
                    )
            provider_tool_results.append(provider_tool_result)
            recent_tool_history.append(
                self._tool_history_entry(
                    tool_call=tool_call,
                    official_result=tool_result,
                    provider_result=provider_tool_result,
                    skipped_by_mentor=False,
                )
            )
            history_events = self._append_agent_observed_tool_events(
                history_events,
                tool_call=tool_call,
                tool_result=provider_tool_result,
                skipped_by_mentor=False,
            )

        return (
            trajectory_tool_calls,
            tool_results,
            provider_tool_results,
            synthetic_contexts,
        )

    def _schedule_agent_tool_calls_with_mentor(
        self,
        tool_calls: list[ToolCall],
        *,
        tick_id: int,
    ) -> None:
        """Start mentored agent tool execution without blocking the audio tick."""
        if self._agent_tool_executor is None:
            raise RuntimeError("Realtime tool-wait executor is not initialized")

        recent_transcript = self._recent_transcript()
        recent_tool_history = self._recent_agent_tool_history()
        history_events = self._recent_agent_observed_history_events()
        pre_gate_by_call_id = {
            tool_call.id: (
                self.tool_mentor is not None
                and self.tool_mentor.should_pre_gate(tool_call.name)
            )
            for tool_call in tool_calls
        }
        for tool_call in tool_calls:
            started_perf = time.perf_counter()
            pre_gate = pre_gate_by_call_id[tool_call.id]
            if pre_gate:
                future = self._agent_tool_executor.submit(
                    self._execute_agent_write_gate_with_mentor_context,
                    tool_call=deepcopy(tool_call),
                    recent_transcript=deepcopy(recent_transcript),
                    recent_tool_history=deepcopy(recent_tool_history),
                    history_events=deepcopy(history_events),
                )
                kind: Literal["execute", "pre_write_gate"] = "pre_write_gate"
                executor_name: Literal["parallel"] = "parallel"
            else:
                future = self._agent_tool_executor.submit(
                    self._execute_agent_tool_calls_with_mentor_context_job,
                    tool_calls=[deepcopy(tool_call)],
                    recent_transcript=deepcopy(recent_transcript),
                    recent_tool_history=deepcopy(recent_tool_history),
                    history_events=deepcopy(history_events),
                )
                kind = "execute"
                executor_name = "parallel"
            job = _PendingAgentToolJob(
                future=future,
                tool_calls=[deepcopy(tool_call)],
                started_tick_id=tick_id,
                started_perf=started_perf,
                kind=kind,
                executor_name=executor_name,
            )
            self._pending_agent_tool_jobs.append(job)
            self._voice_timing_trace["async_tool_jobs"].append(
                {
                    "event": "scheduled",
                    "kind": kind,
                    "executor": executor_name,
                    "worker_count": self._agent_tool_workers,
                    "tick_id": tick_id,
                    "tool_call_ids": [tool_call.id],
                    "tool_names": [tool_call.name],
                    "pending_job_count": len(self._pending_agent_tool_jobs),
                }
            )

    def _execute_agent_tool_calls_with_mentor_context_job(
        self,
        *,
        tool_calls: list[ToolCall],
        recent_transcript: list[dict[str, str]],
        recent_tool_history: list[dict[str, Any]],
        history_events: list[dict[str, Any]],
    ) -> _AgentToolJobResult:
        (
            trajectory_tool_calls,
            tool_results,
            provider_results,
            synthetic_contexts,
        ) = self._execute_agent_tool_calls_with_mentor_context(
            tool_calls=tool_calls,
            recent_transcript=recent_transcript,
            recent_tool_history=recent_tool_history,
            history_events=history_events,
        )
        return _AgentToolJobResult(
            trajectory_tool_calls=trajectory_tool_calls,
            tool_results=tool_results,
            provider_results=provider_results,
            synthetic_contexts=synthetic_contexts,
            approved_write_calls=[],
            approved_write_notes={},
        )

    def _execute_agent_write_gate_with_mentor_context(
        self,
        *,
        tool_call: ToolCall,
        recent_transcript: list[dict[str, str]],
        recent_tool_history: list[dict[str, Any]],
        history_events: list[dict[str, Any]],
    ) -> _AgentToolJobResult:
        """Run only the mentor gate for a mutating tool call."""
        if self.tool_mentor is None:
            return _AgentToolJobResult([], [], [], [], [tool_call], {})

        decision, event, _context = self.tool_mentor.pre_write_gate(
            tool_call=tool_call,
            recent_transcript=recent_transcript,
            recent_tool_history=recent_tool_history,
            history_events=history_events,
        )
        self._voice_timing_trace["mentor_events"].append(event.to_trace_dict())

        if decision.decision == "block":
            block_note = decision.agent_note or "Tool call blocked by mentor."
            provider_result = self._make_mentor_blocked_tool_result(
                tool_call=tool_call,
                block_note=block_note,
            )
            self._record_provider_visible_tool_result(
                tool_call=tool_call,
                kind="mentor_blocked_tool_call",
                official_result_present=False,
                provider_result=provider_result,
            )
            self._voice_timing_trace["tool_events"].append(
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "requestor": tool_call.requestor,
                    "tool_call_to_tool_result_ms": 0.0,
                    "error": False,
                    "skipped_by_mentor": True,
                }
            )
            return _AgentToolJobResult(
                trajectory_tool_calls=[],
                tool_results=[],
                provider_results=[provider_result],
                synthetic_contexts=[],
                approved_write_calls=[],
                approved_write_notes={},
            )

        approved_write_notes = {}
        if decision.agent_note:
            approved_write_notes[tool_call.id] = decision.agent_note
        return _AgentToolJobResult(
            trajectory_tool_calls=[],
            tool_results=[],
            provider_results=[],
            synthetic_contexts=[],
            approved_write_calls=[tool_call],
            approved_write_notes=approved_write_notes,
        )

    @staticmethod
    def _make_mentor_blocked_tool_result(
        *,
        tool_call: ToolCall,
        block_note: str,
    ) -> ToolMessage:
        blocked_content = json.dumps(
            {
                "type": "mentor_blocked_tool_call",
                "agent_note": block_note,
            }
        )
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content=blocked_content,
            requestor=tool_call.requestor,
            error=False,
        )

    def _record_provider_visible_tool_result(
        self,
        *,
        tool_call: ToolCall,
        kind: str,
        official_result_present: bool,
        provider_result: ToolMessage,
        scheduled_tick_id: Optional[int] = None,
        delivered_tick_id: Optional[int] = None,
    ) -> None:
        """Record the exact tool result payload delivered to the agent."""
        event = {
            "tool_name": tool_call.name,
            "tool_call_id": tool_call.id,
            "requestor": tool_call.requestor,
            "kind": kind,
            "official_result_present": official_result_present,
            "content": provider_result.content,
            "error": provider_result.error,
            "scheduled_tick_id": scheduled_tick_id,
            "delivered_tick_id": delivered_tick_id,
        }
        self._voice_timing_trace["provider_tool_results"].append(event)
        self._voice_timing_trace["provider_visible_events"].append(
            {
                "type": "tool_result",
                **event,
            }
        )

    def _collect_completed_agent_tool_jobs(
        self,
        tick_id: int,
        allow_termination: bool = True,
    ) -> tuple[list[ToolMessage], list[dict]]:
        """Collect completed background tool jobs without head-of-line blocking."""
        completed_provider_results: list[ToolMessage] = []
        completed_synthetic_contexts: list[dict] = []

        ready_jobs: list[_PendingAgentToolJob] = []
        still_pending: list[_PendingAgentToolJob] = []
        for job in self._pending_agent_tool_jobs:
            if job.future.done():
                ready_jobs.append(job)
            else:
                still_pending.append(job)
        self._pending_agent_tool_jobs = still_pending

        for job in ready_jobs:
            elapsed_ms = (time.perf_counter() - job.started_perf) * 1000
            job_result = job.future.result()
            if job_result.approved_write_calls:
                (
                    write_tool_calls,
                    write_tool_results,
                    write_provider_results,
                ) = self._execute_approved_agent_write_calls(
                    job_result.approved_write_calls,
                    executed_tick_id=tick_id,
                    approved_write_notes=job_result.approved_write_notes,
                )
                job_result.trajectory_tool_calls.extend(write_tool_calls)
                job_result.tool_results.extend(write_tool_results)
                job_result.provider_results.extend(write_provider_results)
                if allow_termination:
                    self._terminate_if_stop_tool_executed(write_tool_calls)
            self._attach_completed_agent_tool_job_to_trajectory(
                job=job,
                trajectory_tool_calls=job_result.trajectory_tool_calls,
                tool_results=job_result.tool_results,
            )
            completed_provider_results.extend(job_result.provider_results)
            completed_synthetic_contexts.extend(job_result.synthetic_contexts)
            self._voice_timing_trace["async_tool_jobs"].append(
                {
                    "event": "completed",
                    "kind": job.kind,
                    "executor": job.executor_name,
                    "scheduled_tick_id": job.started_tick_id,
                    "executed_tick_id": (
                        tick_id if job_result.approved_write_calls else None
                    ),
                    "delivered_tick_id": tick_id,
                    "elapsed_ms": elapsed_ms,
                    "waited_ticks": tick_id - job.started_tick_id,
                    "tool_call_ids": [tool_call.id for tool_call in job.tool_calls],
                    "tool_names": [tool_call.name for tool_call in job.tool_calls],
                    "approved_write_count": len(job_result.approved_write_calls),
                    "official_result_count": len(job_result.tool_results),
                    "provider_result_count": len(job_result.provider_results),
                    "pending_job_count": len(self._pending_agent_tool_jobs),
                }
            )

        return (
            completed_provider_results,
            completed_synthetic_contexts,
        )

    def _execute_approved_agent_write_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        executed_tick_id: int,
        approved_write_notes: Optional[dict[str, str]] = None,
    ) -> tuple[list[ToolCall], list[ToolMessage], list[ToolMessage]]:
        """Execute mentor-approved writes on the orchestrator tick thread."""
        trajectory_tool_calls: list[ToolCall] = []
        tool_results: list[ToolMessage] = []
        provider_tool_results: list[ToolMessage] = []
        approved_write_notes = approved_write_notes or {}
        for tool_call in tool_calls:
            tool_start = time.perf_counter()
            tool_result = self.environment.get_response(tool_call)
            tool_elapsed_ms = (time.perf_counter() - tool_start) * 1000
            if tool_result.error:
                self.num_errors += 1
            trajectory_tool_calls.append(tool_call)
            tool_results.append(tool_result)
            provider_tool_result = tool_result
            approved_write_note = approved_write_notes.get(tool_call.id)
            if approved_write_note and self.tool_mentor is not None:
                provider_tool_result = self.tool_mentor.augment_tool_result_with_note(
                    tool_result=tool_result,
                    agent_note=approved_write_note,
                )
                self._record_provider_visible_tool_result(
                    tool_call=tool_call,
                    kind="mentor_augmented_write_result",
                    official_result_present=True,
                    provider_result=provider_tool_result,
                )
            provider_tool_results.append(provider_tool_result)
            self._voice_timing_trace["tool_events"].append(
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "requestor": tool_call.requestor,
                    "tool_call_to_tool_result_ms": tool_elapsed_ms,
                    "error": tool_result.error,
                    "executed_after_realtime_wait": True,
                    "executed_tick_id": executed_tick_id,
                }
            )
        return trajectory_tool_calls, tool_results, provider_tool_results

    def _attach_completed_agent_tool_job_to_trajectory(
        self,
        *,
        job: _PendingAgentToolJob,
        trajectory_tool_calls: list[ToolCall],
        tool_results: list[ToolMessage],
    ) -> None:
        """Attach async tool results to the original tool-call tick for replay."""
        if job.started_tick_id >= len(self.ticks):
            logger.warning(
                "Completed async tool job before its scheduling tick was recorded"
            )
            return

        tick = self.ticks[job.started_tick_id]
        job_call_ids = {tool_call.id for tool_call in job.tool_calls}
        trajectory_by_id = {
            tool_call.id: tool_call for tool_call in trajectory_tool_calls
        }
        existing_calls_by_id = {
            tool_call.id: tool_call for tool_call in tick.agent_tool_calls
        }

        existing_results_by_id = {
            result.id: result
            for result in tick.agent_tool_results
            if result.id not in job_call_ids
        }
        new_results_by_id = {result.id: result for result in tool_results}
        completed_call_ids = [
            result.id
            for result in tick.agent_tool_results
            if result.id not in job_call_ids
        ]
        completed_call_ids.extend(result.id for result in tool_results)

        ordered_calls: list[ToolCall] = []
        emitted_call_ids: set[str] = set()
        for call_id in completed_call_ids:
            tool_call = trajectory_by_id.get(call_id) or existing_calls_by_id.get(
                call_id
            )
            if tool_call is not None and call_id not in emitted_call_ids:
                ordered_calls.append(deepcopy(tool_call))
                emitted_call_ids.add(call_id)
        for existing_call in tick.agent_tool_calls:
            if existing_call.id in emitted_call_ids:
                continue
            if (
                existing_call.id in job_call_ids
                and existing_call.id not in trajectory_by_id
            ):
                continue
            replacement = trajectory_by_id.get(existing_call.id, existing_call)
            ordered_calls.append(deepcopy(replacement))
            emitted_call_ids.add(existing_call.id)
        for tool_call in trajectory_tool_calls:
            if tool_call.id not in emitted_call_ids:
                ordered_calls.append(deepcopy(tool_call))
                emitted_call_ids.add(tool_call.id)
        tick.agent_tool_calls = ordered_calls

        emitted_result_ids: set[str] = set()
        ordered_results: list[ToolMessage] = []
        for tool_call in tick.agent_tool_calls:
            result = new_results_by_id.get(tool_call.id) or existing_results_by_id.get(
                tool_call.id
            )
            if result is not None:
                ordered_results.append(deepcopy(result))
                emitted_result_ids.add(tool_call.id)
        for result in tick.agent_tool_results:
            if result.id not in emitted_result_ids and result.id not in job_call_ids:
                ordered_results.append(result)
                emitted_result_ids.add(result.id)
        for result in tool_results:
            if result.id not in emitted_result_ids:
                ordered_results.append(deepcopy(result))
        tick.agent_tool_results = ordered_results

    def _remove_pending_agent_tool_job_from_trajectory(
        self,
        job: _PendingAgentToolJob,
    ) -> None:
        """Remove an undelivered async tool call from executable replay."""
        if job.started_tick_id >= len(self.ticks):
            return
        tick = self.ticks[job.started_tick_id]
        job_call_ids = {tool_call.id for tool_call in job.tool_calls}
        tick.agent_tool_calls = [
            tool_call
            for tool_call in tick.agent_tool_calls
            if tool_call.id not in job_call_ids
        ]
        tick.agent_tool_results = [
            result
            for result in tick.agent_tool_results
            if result.id not in job_call_ids
        ]

    def _serialize_synthetic_context(
        self,
        context: SyntheticAgentContext,
    ) -> dict:
        """Create runtime and trace payloads for a mentor context."""
        data = context.to_trace_dict()
        data["content"] = context.to_model_context()
        return data

    def _recent_transcript(
        self,
        max_items: int = 16,
        max_chars: int = 6000,
    ) -> list[dict[str, str]]:
        """Build compact turn-like transcript text from streaming chunks."""
        items: list[dict[str, str]] = []
        current_role: Optional[str] = None
        current_parts: list[str] = []

        def flush_current() -> None:
            nonlocal current_role, current_parts
            if current_role is not None and current_parts:
                text = "".join(current_parts).strip()
                if text:
                    items.append({"role": current_role, "text": text})
            current_role = None
            current_parts = []

        def append_part(role: str, text: str) -> None:
            nonlocal current_role
            if current_role != role:
                flush_current()
                current_role = role
            if role == "user" and current_parts:
                current_parts.append(" ")
            current_parts.append(text)

        for tick in self.ticks:
            for transcript in self._agent_observed_input_transcripts(tick.agent_chunk):
                append_part("user", transcript)
            if tick.agent_chunk and tick.agent_chunk.content:
                append_part("assistant", tick.agent_chunk.content)
        flush_current()

        compact = items[-max_items:]
        while (
            len(compact) > 1 and sum(len(item["text"]) for item in compact) > max_chars
        ):
            compact.pop(0)
        if compact and sum(len(item["text"]) for item in compact) > max_chars:
            compact[0]["text"] = compact[0]["text"][-max_chars:]
        return compact

    @staticmethod
    def _agent_observed_input_transcripts(
        agent_chunk: Optional[AssistantMessage],
    ) -> list[str]:
        """Return user transcripts emitted by the agent provider."""
        return [
            str(item["transcript"])
            for item in FullDuplexOrchestrator._agent_observed_input_transcript_items(
                agent_chunk
            )
            if item.get("transcript")
        ]

    @staticmethod
    def _agent_observed_input_transcript_items(
        agent_chunk: Optional[AssistantMessage],
    ) -> list[dict[str, Any]]:
        """Return user transcript records emitted by the agent provider."""
        if agent_chunk is None or not isinstance(agent_chunk.raw_data, dict):
            return []

        raw_transcripts = agent_chunk.raw_data.get("input_audio_transcripts") or []
        transcripts: list[dict[str, Any]] = []
        for item in raw_transcripts:
            if isinstance(item, dict):
                if item.get("transcript"):
                    transcripts.append(dict(item))
            else:
                if item:
                    transcripts.append({"transcript": str(item)})
        return transcripts

    def _recent_agent_tool_history(self) -> list[dict[str, Any]]:
        """Build compact full-history agent tool call/result records."""
        history: list[dict[str, Any]] = []
        results_by_id: dict[str, ToolMessage] = {}
        for tick in self.ticks:
            for result in tick.agent_tool_results or []:
                results_by_id[result.id] = result

        for tick in self.ticks:
            calls = tick.agent_tool_calls or []
            for tool_call in calls:
                history.append(
                    self._tool_history_entry(
                        tool_call=tool_call,
                        official_result=results_by_id.get(tool_call.id),
                        provider_result=None,
                        skipped_by_mentor=False,
                    )
                )
        return history

    def _recent_agent_observed_history_events(self) -> list[dict[str, Any]]:
        """Build readable agent-observed history with turn-like transcript events."""
        provider_results_by_id: dict[str, dict[str, Any]] = {}
        for item in self._voice_timing_trace.get("provider_tool_results", []):
            call_id = item.get("tool_call_id")
            if call_id:
                provider_results_by_id[str(call_id)] = item

        events: list[dict[str, Any]] = []
        emitted_result_ids: set[str] = set()
        current_role: Optional[str] = None
        current_parts: list[str] = []

        def flush_current() -> None:
            nonlocal current_role, current_parts
            if current_role is not None and current_parts:
                text = "".join(current_parts).strip()
                if text:
                    if current_role == "user":
                        events.append(
                            {
                                "kind": "customer_speech_transcript",
                                "source": "agent_observed_stt_turn_merged",
                                "text": text,
                            }
                        )
                    else:
                        events.append({"kind": "voice_agent_speech", "text": text})
            current_role = None
            current_parts = []

        def append_turn_part(role: str, text: str) -> None:
            nonlocal current_role
            if current_role != role:
                flush_current()
                current_role = role
            if role == "user" and current_parts:
                current_parts.append(" ")
            current_parts.append(text)

        for tick in self.ticks:
            for transcript_item in self._agent_observed_input_transcript_items(
                tick.agent_chunk
            ):
                transcript = transcript_item.get("transcript")
                if transcript:
                    append_turn_part("user", str(transcript))
            if tick.agent_chunk and tick.agent_chunk.content:
                append_turn_part("assistant", tick.agent_chunk.content)
            results_by_id = {
                result.id: result for result in tick.agent_tool_results or []
            }
            for tool_call in tick.agent_tool_calls or []:
                flush_current()
                events.append(self._agent_observed_tool_call_event(tool_call))
                provider_trace = provider_results_by_id.get(tool_call.id)
                official_result = results_by_id.get(tool_call.id)
                if provider_trace is not None:
                    events.append(
                        {
                            "kind": "voice_agent_visible_tool_result",
                            "tool_call_id": tool_call.id,
                            "content": self._truncate_tool_history_content(
                                str(provider_trace.get("content") or "")
                            ),
                            "error": False,
                            "content_source": "provider_visible",
                            "skipped_by_mentor": provider_trace.get("kind")
                            == "mentor_blocked_tool_call",
                        }
                    )
                    emitted_result_ids.add(tool_call.id)
                elif official_result is not None:
                    events.append(
                        self._agent_observed_tool_result_event(
                            official_result,
                            skipped_by_mentor=False,
                            content_source="official",
                        )
                    )
                    emitted_result_ids.add(tool_call.id)
            for tool_result in tick.agent_tool_results or []:
                if tool_result.id in emitted_result_ids:
                    continue
                flush_current()
                provider_trace = provider_results_by_id.get(tool_result.id)
                if provider_trace is not None:
                    events.append(
                        {
                            "kind": "voice_agent_visible_tool_result",
                            "tool_call_id": tool_result.id,
                            "content": self._truncate_tool_history_content(
                                str(provider_trace.get("content") or "")
                            ),
                            "error": False,
                            "content_source": "provider_visible",
                            "skipped_by_mentor": provider_trace.get("kind")
                            == "mentor_blocked_tool_call",
                        }
                    )
                else:
                    events.append(
                        self._agent_observed_tool_result_event(
                            tool_result,
                            skipped_by_mentor=False,
                            content_source="official",
                        )
                    )
                emitted_result_ids.add(tool_result.id)
            for context in tick.agent_synthetic_context or []:
                content = context.get("content")
                if content:
                    flush_current()
                    events.append(
                        {
                            "kind": "mentor_context",
                            "content": str(content),
                            "parent_tool_call_id": context.get("parent_tool_call_id"),
                        }
                    )
        flush_current()
        return self._renumber_agent_observed_history_events(events)

    def _append_agent_observed_tool_events(
        self,
        history_events: list[dict[str, Any]],
        *,
        tool_call: ToolCall,
        tool_result: ToolMessage,
        skipped_by_mentor: bool,
    ) -> list[dict[str, Any]]:
        events = [dict(item) for item in history_events]
        events.append(self._agent_observed_tool_call_event(tool_call))
        events.append(
            self._agent_observed_tool_result_event(
                tool_result,
                skipped_by_mentor=skipped_by_mentor,
                content_source="provider_visible",
            )
        )
        return self._renumber_agent_observed_history_events(events)

    @staticmethod
    def _agent_observed_tool_call_event(tool_call: ToolCall) -> dict[str, Any]:
        return {
            "kind": "voice_agent_tool_call",
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
        }

    def _agent_observed_tool_result_event(
        self,
        tool_result: ToolMessage,
        *,
        skipped_by_mentor: bool,
        content_source: str,
    ) -> dict[str, Any]:
        return {
            "kind": "voice_agent_visible_tool_result",
            "tool_call_id": tool_result.id,
            "content": self._truncate_tool_history_content(tool_result.content),
            "error": bool(tool_result.error),
            "content_source": content_source,
            "skipped_by_mentor": skipped_by_mentor,
        }

    @staticmethod
    def _renumber_agent_observed_history_events(
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        numbered: list[dict[str, Any]] = []
        for seq, event in enumerate(events, start=1):
            clean = dict(event)
            clean["seq"] = seq
            numbered.append(clean)
        return numbered

    def _tool_history_entry(
        self,
        *,
        tool_call: ToolCall,
        official_result: Optional[ToolMessage],
        provider_result: Optional[ToolMessage],
        skipped_by_mentor: bool,
    ) -> dict[str, Any]:
        """Serialize one agent tool boundary event for mentor context."""
        entry: dict[str, Any] = {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
            "skipped_by_mentor": skipped_by_mentor,
        }
        if official_result is not None:
            entry["official_result"] = self._truncate_tool_history_content(
                official_result.content
            )
            entry["official_error"] = official_result.error
        if provider_result is not None:
            entry["provider_result"] = self._truncate_tool_history_content(
                provider_result.content
            )
            entry["provider_error"] = provider_result.error
        return entry

    def _truncate_tool_history_content(self, content: Optional[str]) -> str:
        limit = (
            self.tool_mentor.config.tool_mentor_max_result_chars
            if self.tool_mentor is not None
            else 6000
        )
        return (content or "")[:limit]

    def get_trajectory(self) -> list[Tick]:
        """
        Get the tick-based trajectory of the simulation.

        Returns:
            List of Tick objects, each containing all events from a simulation tick.
        """
        return deepcopy(self.ticks)

    def get_ticks(self) -> list[Tick]:
        """
        Get the tick-based trajectory.

        Alias for get_trajectory() for backward compatibility.

        Returns:
            List of Tick objects, each containing all events from a simulation tick.
        """
        return self.get_trajectory()

    def get_messages(self) -> list[Message]:
        """
        Get all messages from the simulation as a flat list.

        Converts the tick-based trajectory to a flat message list,
        sorted by timestamp with turn_idx assigned.

        Returns:
            List of all messages sorted by timestamp with turn_idx assigned.
        """
        messages: list[Message] = []
        for tick in self.ticks:
            messages.extend(tick.get_all_messages())
        messages = sorted(messages, key=lambda m: m.timestamp)
        result = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            result.append(msg)
        return result

    def _check_termination(self) -> None:
        """
        Check for full-duplex specific termination conditions.

        Checks max_steps, max_errors, and timeout after each tick.
        """
        if self.step_count >= self.max_steps:
            self.done = True
            self.termination_reason = TerminationReason.MAX_STEPS
        if self.num_errors >= self.max_errors:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        self._check_timeout()

    def _shutdown_agent_tool_executor(self) -> None:
        """Stop background tool execution resources."""
        if self._agent_tool_executor is None:
            return
        if self._pending_agent_tool_jobs:
            self._voice_timing_trace["async_tool_jobs"].append(
                {
                    "event": "shutdown_with_pending_jobs",
                    "pending_job_count": len(self._pending_agent_tool_jobs),
                    "tool_call_ids": [
                        tool_call.id
                        for job in self._pending_agent_tool_jobs
                        for tool_call in job.tool_calls
                    ],
                }
            )
        self._agent_tool_executor.shutdown(wait=False, cancel_futures=True)
        self._agent_tool_executor = None

    def _expire_pending_agent_tool_jobs_for_finalize(self) -> None:
        """Expire in-flight tool jobs when the conversation has ended."""
        while self._pending_agent_tool_jobs:
            job = self._pending_agent_tool_jobs[0]
            self._pending_agent_tool_jobs.pop(0)
            was_done = job.future.done()
            cancelled = job.future.cancel()
            self._remove_pending_agent_tool_job_from_trajectory(job)
            self._voice_timing_trace["async_tool_jobs"].append(
                {
                    "event": "expired_after_conversation_end",
                    "kind": job.kind,
                    "executor": job.executor_name,
                    "scheduled_tick_id": job.started_tick_id,
                    "elapsed_ms": (time.perf_counter() - job.started_perf) * 1000,
                    "future_done": was_done,
                    "cancelled": cancelled,
                    "tool_call_ids": [tool_call.id for tool_call in job.tool_calls],
                    "tool_names": [tool_call.name for tool_call in job.tool_calls],
                    "official_result_count": 0,
                    "provider_result_count": 0,
                    "synthetic_context_count": 0,
                }
            )

    def _drain_pending_agent_tool_jobs_for_finalize(self) -> None:
        """Attach terminal realtime tool jobs before final scoring.

        In realtime-wait mode a tool call can be scheduled on the same tick that
        ends the conversation. The normal next-tick collector will not run after
        that, so finalization must attach completed terminal jobs to the
        original tick before expiring any remaining jobs.
        """
        if not self._pending_agent_tool_jobs:
            return

        final_tick_id = len(self.ticks)
        self._collect_completed_agent_tool_jobs(
            final_tick_id, allow_termination=False
        )

        still_pending: list[_PendingAgentToolJob] = []
        for job in self._pending_agent_tool_jobs:
            if not job.future.done():
                try:
                    job.future.result(timeout=10.0)
                except TimeoutError:
                    still_pending.append(job)
                    continue
                except Exception:
                    still_pending.append(job)
                    continue
            if job.future.cancelled():
                still_pending.append(job)
                continue
            if job.future.done() and job.future.exception() is not None:
                still_pending.append(job)
                continue
            still_pending.append(job)
        self._pending_agent_tool_jobs = still_pending

        self._collect_completed_agent_tool_jobs(
            final_tick_id, allow_termination=False
        )

    def _cleanup(self) -> None:
        """Clean up participant and background tool resources after exceptions."""
        try:
            self._shutdown_agent_tool_executor()
        finally:
            super()._cleanup()

    def _finalize(self) -> SimulationRun:
        """
        Finalize the full-duplex simulation and create the SimulationRun result.

        Sends stop signals to agent and user, calculates costs, and builds the result.

        Returns:
            SimulationRun with all simulation data.
        """
        # Send stop signals to agent and user, forwarding any pending tool results
        try:
            self.agent.stop(
                None, self.agent_state, tool_results=self.pending_agent_tool_results
            )
        except Exception as e:
            logger.warning(f"Error stopping agent during finalization: {e}")
        try:
            self.user.stop(
                None, self.user_state, tool_results=self.pending_user_tool_results
            )
        except Exception as e:
            logger.warning(f"Error stopping user during finalization: {e}")
        self._drain_pending_agent_tool_jobs_for_finalize()
        self._expire_pending_agent_tool_jobs_for_finalize()
        self._shutdown_agent_tool_executor()

        # Calculate duration
        duration = time.perf_counter() - self._run_start_perf

        # Derive messages for cost computation (not stored in SimulationRun)
        ticks = self.get_trajectory()
        res = get_cost(self.get_messages())
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res

        # Get speech_environment from user's voice_settings if available
        speech_environment = None
        if (
            hasattr(self.user, "voice_settings")
            and self.user.voice_settings is not None
        ):
            speech_environment = self.user.voice_settings.speech_environment

        # Compute responsiveness metrics
        info = compute_responsiveness_info(ticks)
        if info is None:
            info = {}
        info["voice_timing_trace"] = self._voice_timing_trace

        # Extract provider session ID if available (e.g., OpenAI session ID)
        provider_session_id = None
        if hasattr(self.agent, "adapter") and hasattr(self.agent.adapter, "provider"):
            provider = self.agent.adapter.provider
            provider_session_id = getattr(provider, "session_id", None)

        # Collect effect timeline from user simulator if available
        effect_timeline = None
        if hasattr(self.user, "get_effect_timeline"):
            effect_timeline = self.user.get_effect_timeline()
            if effect_timeline and effect_timeline.events:
                logger.info(
                    f"Effect timeline: {len(effect_timeline.events)} events recorded"
                )

        simulation_run = SimulationRun(
            id=self.simulation_id,
            task_id=self.task.id,
            start_time=self._run_start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            ticks=ticks,
            seed=self.seed,
            mode=self.mode.value,
            speech_environment=speech_environment,
            info=info,
            provider_session_id=provider_session_id,
            effect_timeline=effect_timeline,
        )
        return simulation_run

    def run(self) -> SimulationRun:
        """
        Run the simulation with post-processing for user transcripts.

        Overrides the base class run() to track timing and add transcript processing.

        Returns:
            SimulationRun: The simulation run.
        """
        result = super().run()
        compute_proportional_user_transcripts(self.ticks)
        return result
