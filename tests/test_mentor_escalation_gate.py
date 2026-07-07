"""Escalation-gate and mentor-packet guardrails.

transfer_to_human_agents is GENERIC (neither a read nor a write tool), so
without explicit wiring it bypasses every mentor hook while still terminating
the conversation. These tests pin that wiring: configured escalation tools
route through the pre-execution gate, the mentor packet carries the agent
tool inventory, and the prompt contracts (hand-off clause, uncertainty
marking, tool-reference rule) stay present.
"""

import json

import tau2.voice.audio_native.mentor.tool_boundary_mentor as tbm
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.environment.toolkit import ToolType
from tau2.registry import registry


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.refusal = None


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _mentor(**overrides):
    env = registry.get_env_constructor("telecom")()
    cfg = AudioNativeConfig(
        provider="openai",
        tool_mentor_enabled=True,
        tool_mentor_mode="llm",
        tool_mentor_model="gemini/gemini-3.5-flash",
        tool_mentor_reasoning_effort="low",
        **overrides,
    )
    return tbm.ToolBoundaryMentor(config=cfg, environment=env)


def _blocks(user_message):
    content = user_message["content"]
    history = content.split("<history_block>\n", 1)[1].split("\n</history_block>", 1)[0]
    current = content.split("<current_tool_event>\n", 1)[1].split(
        "\n</current_tool_event>", 1
    )[0]
    return json.loads(history), json.loads(current)


TRANSFER_CALL = ToolCall(
    id="c_esc",
    name="transfer_to_human_agents",
    arguments={"summary": "Customer needs help beyond self-service."},
    requestor="assistant",
)


def test_escalation_tool_routes_through_pre_gate_by_default():
    mentor = _mentor()
    assert mentor.is_escalation_tool("transfer_to_human_agents") is True
    assert mentor.should_pre_gate("transfer_to_human_agents") is True
    assert mentor.should_post_read_note("transfer_to_human_agents") is False
    assert mentor.should_pre_gate("refuel_data") is True
    assert mentor.should_pre_gate("get_customer_by_phone") is False


def test_empty_escalation_list_disables_gating():
    mentor = _mentor(tool_mentor_escalation_tools=[])
    assert mentor.is_escalation_tool("transfer_to_human_agents") is False
    assert mentor.should_pre_gate("transfer_to_human_agents") is False


def test_escalation_gate_block_decision_and_handoff_event_type(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _FakeResponse(
            json.dumps(
                {
                    "decision": "block",
                    "agent_note": "Untried remedies remain; use them first.",
                    "missing_requirements": ["untried_agent_remedy"],
                }
            )
        )

    monkeypatch.setattr(tbm, "completion", fake_completion)
    mentor = _mentor()
    decision, event, context = mentor.pre_write_gate(
        TRANSFER_CALL, recent_transcript=[]
    )
    assert decision.decision == "block"
    assert event.stage == "pre_write"
    assert event.tool_type == ToolType.GENERIC.value
    assert context is not None and context.agent_note
    _, current_tool_event = _blocks(calls[0]["messages"][1])
    assert current_tool_event["type"] == "proposed_handoff_tool_call"


def test_escalation_gate_allow_decision(monkeypatch):
    def fake_completion(**kwargs):
        return _FakeResponse(
            json.dumps(
                {
                    "decision": "allow",
                    "agent_note": "",
                    "missing_requirements": [],
                }
            )
        )

    monkeypatch.setattr(tbm, "completion", fake_completion)
    decision, _event, context = _mentor().pre_write_gate(
        TRANSFER_CALL, recent_transcript=[]
    )
    assert decision.decision == "allow"
    assert context is None


def test_mentor_packet_lists_agent_tool_inventory(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(json.dumps({"agent_note": ""}))

    monkeypatch.setattr(tbm, "completion", fake_completion)
    mentor = _mentor()
    tool_call = ToolCall(
        id="c_read",
        name="get_customer_by_phone",
        arguments={"phone_number": "555-123-2002"},
        requestor="assistant",
    )
    tool_result = ToolMessage(
        id="c_read", role="tool", requestor="assistant", content="{}", error=False
    )
    mentor._llm_post_read(tool_call, tool_result, [])

    payload, _ = _blocks(captured["messages"][1])
    assert payload["history_block_version"] == "agent_observed_history_v3"
    listed = {tool["name"] for tool in payload["available_tools"]}
    env = registry.get_env_constructor("telecom")()
    assert listed == set(env.tools.tools)
    assert all(tool["description"] for tool in payload["available_tools"])


def test_prompt_contract_sentences_present():
    """Regression pin that structural prompt contracts survive refactors.

    This is not a content-boundary check (the prompt constitution
    deliberately has no static litmus test; content review is human); it
    only pins that the tool-reference and uncertainty-marking contracts
    still exist in the prompt.
    """
    mentor = _mentor()
    mentor._policy_text = lambda: ""
    pre = mentor._system_prompt("pre_write")
    post = mentor._system_prompt("post_read")
    for prompt in (pre, post):
        assert "available_tools" in prompt
        assert "mark it as a guess" in prompt
    assert "Hand-off or give-up actions" in pre
