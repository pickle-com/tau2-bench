from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import (
    AgentInfo,
    AudioNativeConfig,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.metrics.agent_metrics import compute_metrics
from tau2.orchestrator.modes import CommunicationMode
from tau2.scripts import evaluate_trajectories as evaluate_trajectories_script
from tau2.scripts.leaderboard.prepare_submission import (
    _detect_voice_mode,
    validate_submission_traj_set,
)


def _voice_results_with_mentor_context(
    domain: str = "retail",
    tool_mentor_enabled: bool = True,
) -> Results:
    task = Task(
        id="task_1",
        user_scenario=UserScenario(instructions="Find available variants."),
        evaluation_criteria=EvaluationCriteria(),
    )
    tick = Tick(
        tick_id=1,
        timestamp="2026-01-01T00:00:01",
        agent_chunk=AssistantMessage(
            role="assistant",
            content=None,
            contains_speech=False,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_product_details",
                    arguments={"product_id": "p1"},
                )
            ],
        ),
        agent_tool_results=[
            ToolMessage(
                role="tool",
                id="call_1",
                content='{"variants":{"v1":{"available":true}}}',
            )
        ],
        agent_synthetic_context=[
            {
                "type": "mentor_note",
                "stage": "post_read",
                "content": "Available variants: 1.",
            }
        ],
    )
    simulation = SimulationRun(
        id="sim_1",
        task_id="task_1",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:02",
        duration=2.0,
        termination_reason=TerminationReason.USER_STOP,
        reward_info=RewardInfo(reward=1.0),
        messages=None,
        ticks=[tick],
        trial=0,
        seed=123,
        mode=CommunicationMode.FULL_DUPLEX.value,
        info={"voice_timing_trace": {"tool_events": [{"tool_name": "get_product_details"}]}},
    )
    return Results(
        info=Info(
            git_commit="abc123",
            num_trials=1,
            max_steps=100,
            max_errors=3,
            user_info=UserInfo(
                implementation="audio_native_user",
                llm="voice-user-sim-v1",
                llm_args={"speech_complexity": "regular"},
            ),
            agent_info=AgentInfo(
                implementation="audio_native_agent",
                llm="gpt-realtime-1.5",
                llm_args={"tool_mentor_enabled": tool_mentor_enabled},
            ),
            environment_info=EnvironmentInfo(domain_name=domain, policy="policy"),
            speech_complexity="regular",
            audio_native_config=AudioNativeConfig(
                provider="openai",
                model="gpt-realtime-1.5",
                tool_mentor_enabled=tool_mentor_enabled,
                tool_mentor_mode="heuristic",
            ),
        ),
        tasks=[task],
        simulations=[simulation],
    )


def test_voice_mentor_context_round_trips_without_affecting_submission_metrics(
    tmp_path,
):
    results = _voice_results_with_mentor_context()
    results.save(tmp_path / "results.json", format="dir")

    loaded = Results.load(tmp_path / "results.json")
    simulation = loaded.simulations[0]
    tick = simulation.ticks[0]

    assert _detect_voice_mode([loaded]) is True
    valid, error = validate_submission_traj_set([loaded])
    assert valid, error
    assert tick.agent_synthetic_context[0]["content"] == "Available variants: 1."
    assert simulation.info["voice_timing_trace"]["tool_events"][0]["tool_name"] == (
        "get_product_details"
    )

    flattened = simulation.get_messages()
    assert [message.role for message in flattened] == ["assistant", "tool"]
    assert all("Available variants: 1." not in (message.content or "") for message in flattened)

    metrics = compute_metrics(loaded)
    assert metrics.pass_hat_ks[1] == 1.0
    assert metrics.total_simulations == 1


def test_full_duplex_simulation_exposes_tool_replay_messages():
    tick = Tick(
        tick_id=1,
        timestamp="2026-01-01T00:00:01",
        user_chunk=UserMessage(
            role="user",
            content="Can you check that?",
            timestamp="2026-01-01T00:00:02",
        ),
        agent_chunk=AssistantMessage(
            role="assistant",
            content=None,
            contains_speech=False,
        ),
        agent_tool_calls=[
            ToolCall(
                id="call_1",
                name="get_product_details",
                arguments={"product_id": "p1"},
            )
        ],
        agent_tool_results=[
            ToolMessage(
                role="tool",
                id="call_1",
                content='{"ok":true}',
                timestamp="2026-01-01T00:00:03",
            )
        ],
    )
    simulation = SimulationRun(
        id="sim_1",
        task_id="task_1",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:04",
        duration=4.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=None,
        ticks=[tick],
        mode=CommunicationMode.FULL_DUPLEX.value,
    )

    replay = simulation.get_tool_replay_messages()

    assert [message.role for message in replay] == ["assistant", "tool"]
    assert replay[0].tool_calls[0].id == "call_1"
    assert replay[1].id == "call_1"
    assert [message.turn_idx for message in replay] == [0, 1]


def test_compute_simulation_rewards_uses_full_duplex_mode_for_voice_ticks(monkeypatch):
    captured_modes = []

    def _capture_evaluate_simulation(**kwargs):
        captured_modes.append(kwargs["mode"])
        return RewardInfo(reward=1.0)

    monkeypatch.setattr(
        evaluate_trajectories_script,
        "evaluate_simulation",
        _capture_evaluate_simulation,
    )
    results = _voice_results_with_mentor_context()

    updated = evaluate_trajectories_script.compute_simulation_rewards(results)

    assert captured_modes == [CommunicationMode.FULL_DUPLEX]
    assert updated.simulations[0].reward_info.reward == 1.0


def test_submission_validation_rejects_mixed_tool_mentor_config():
    baseline_domain = _voice_results_with_mentor_context(
        domain="retail",
        tool_mentor_enabled=False,
    )
    mentor_domain = _voice_results_with_mentor_context(
        domain="airline",
        tool_mentor_enabled=True,
    )

    valid, error = validate_submission_traj_set([baseline_domain, mentor_domain])

    assert valid is False
    assert "Agent / User Simulator" in error
