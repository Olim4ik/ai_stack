"""Unit tests for agent graph routing logic."""

from services.agent.src.graph.router import route_after_execute, route_by_query_type


class TestRouteByQueryType:
    def test_simple_routes_to_retrieve(self):
        state = {"query_type": "simple"}
        assert route_by_query_type(state) == "retrieve"

    def test_multi_step_routes_to_plan(self):
        state = {"query_type": "multi_step"}
        assert route_by_query_type(state) == "plan"

    def test_action_routes_to_execute(self):
        state = {"query_type": "action"}
        assert route_by_query_type(state) == "execute"

    def test_unknown_defaults_to_retrieve(self):
        state = {"query_type": "unknown"}
        assert route_by_query_type(state) == "retrieve"

    def test_empty_defaults_to_retrieve(self):
        state = {}
        assert route_by_query_type(state) == "retrieve"


class TestRouteAfterExecute:
    def test_pending_confirmation_routes_to_confirm(self):
        state = {"pending_confirmation": {"tool": "jira_create", "description": "test"}}
        assert route_after_execute(state) == "confirm"

    def test_no_confirmation_routes_to_synthesize(self):
        state = {"pending_confirmation": None}
        assert route_after_execute(state) == "synthesize"

    def test_missing_key_routes_to_synthesize(self):
        state = {}
        assert route_after_execute(state) == "synthesize"
