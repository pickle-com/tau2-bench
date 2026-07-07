import pytest

from tau2.data_model.simulation import AudioNativeConfig
from tau2.environment.toolkit import ToolType
from tau2.registry import registry
from tau2.voice.audio_native.mentor.tool_boundary_mentor import ToolBoundaryMentor

EXPECTED_TOOL_COVERAGE = {
    "retail": {
        "read": {
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "get_item_details",
            "get_order_details",
            "get_product_details",
            "get_user_details",
            "list_all_product_types",
        },
        "mutable_write": {
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
        },
        "generic": {"calculate", "transfer_to_human_agents"},
    },
    "airline": {
        "read": {
            "get_flight_status",
            "get_reservation_details",
            "get_user_details",
            "list_all_airports",
            "search_direct_flight",
            "search_onestop_flight",
        },
        "mutable_write": {
            "book_reservation",
            "cancel_reservation",
            "send_certificate",
            "update_reservation_baggages",
            "update_reservation_flights",
            "update_reservation_passengers",
        },
        "generic": {"calculate", "transfer_to_human_agents"},
    },
    "telecom": {
        "read": {
            "get_bills_for_customer",
            "get_customer_by_id",
            "get_customer_by_name",
            "get_customer_by_phone",
            "get_data_usage",
            "get_details_by_id",
        },
        "mutable_write": {
            "disable_roaming",
            "enable_roaming",
            "refuel_data",
            "resume_line",
            "send_payment_request",
            "suspend_line",
        },
        "generic": {"transfer_to_human_agents"},
    },
}


@pytest.mark.parametrize("domain", sorted(EXPECTED_TOOL_COVERAGE))
def test_tool_mentor_routes_all_tau_voice_domain_tools_by_metadata(domain):
    env = registry.get_env_constructor(domain)()
    mentor = ToolBoundaryMentor(
        config=AudioNativeConfig(
            provider="xai",
            tool_mentor_enabled=True,
            tool_mentor_mode="heuristic",
        ),
        environment=env,
    )

    observed = {"read": set(), "mutable_write": set(), "generic": set()}
    other_tools = set()
    for tool_name in env.tools.tools:
        tool_type = env.tools.tool_type(tool_name)
        mutates_state = env.tools.tool_mutates_state(tool_name)
        if tool_type == ToolType.READ and not mutates_state:
            observed["read"].add(tool_name)
        elif tool_type == ToolType.WRITE and mutates_state:
            observed["mutable_write"].add(tool_name)
        elif tool_type == ToolType.GENERIC:
            observed["generic"].add(tool_name)
        else:
            other_tools.add(tool_name)

    assert observed == EXPECTED_TOOL_COVERAGE[domain]
    assert other_tools == set()

    for tool_name in observed["read"]:
        assert mentor.should_post_read_note(tool_name) is True
        assert mentor.should_pre_gate(tool_name) is False
    for tool_name in observed["mutable_write"]:
        assert mentor.should_pre_gate(tool_name) is True
        assert mentor.should_post_read_note(tool_name) is False
    for tool_name in observed["generic"]:
        expected_gate = tool_name == "transfer_to_human_agents"
        assert mentor.should_pre_gate(tool_name) is expected_gate
        assert mentor.should_post_read_note(tool_name) is False
