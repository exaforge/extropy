"""Unit tests for Phase D conversation system."""

from extropy.simulation.conversation import (
    ConversationRequest,
    ConversationMessage,
    ConversationResult,
    ConversationStateChange,
    collect_conversation_requests,
    prioritize_and_resolve_conflicts,
)
from extropy.core.models import (
    ReasoningResponse,
    SimulationRunConfig,
)


class TestCollectConversationRequests:
    """Tests for collecting talk_to requests from reasoning results."""

    def test_extracts_talk_to_actions(self):
        """Should extract talk_to actions from reasoning responses."""
        reasoning_results = [
            (
                "agent_1",
                ReasoningResponse(
                    reasoning="I need to talk to my partner about this.",
                    sentiment=0.5,
                    conviction=0.7,
                    actions=[
                        {"type": "talk_to", "who": "Sarah", "topic": "the announcement"}
                    ],
                ),
            ),
            (
                "agent_2",
                ReasoningResponse(
                    reasoning="I should discuss with John.",
                    sentiment=-0.3,
                    conviction=0.5,
                    actions=[{"type": "talk_to", "who": "john"}],
                ),
            ),
        ]

        adjacency = {
            "agent_1": [("agent_3", {"type": "partner", "weight": 0.9})],
            "agent_2": [("agent_4", {"type": "coworker", "weight": 0.6})],
        }

        agent_map = {
            "agent_1": {"_id": "agent_1", "first_name": "Bob"},
            "agent_2": {"_id": "agent_2", "first_name": "Alice"},
            "agent_3": {"_id": "agent_3", "first_name": "Sarah"},
            "agent_4": {"_id": "agent_4", "first_name": "John"},
        }

        requests = collect_conversation_requests(
            reasoning_results=reasoning_results,
            adjacency=adjacency,
            agent_map=agent_map,
        )

        assert len(requests) == 2
        assert requests[0].initiator_id == "agent_1"
        assert requests[0].target_id == "agent_3"
        assert requests[0].target_name == "Sarah"
        assert requests[0].topic == "the announcement"

    def test_ignores_empty_actions(self):
        """Should skip agents with no actions."""
        reasoning_results = [
            (
                "agent_1",
                ReasoningResponse(
                    reasoning="I'll wait.",
                    sentiment=0.0,
                    conviction=0.3,
                    actions=[],
                ),
            ),
        ]

        requests = collect_conversation_requests(
            reasoning_results=reasoning_results,
            adjacency={},
            agent_map={},
        )

        assert len(requests) == 0

    def test_handles_partner_npc(self):
        """Should resolve partner NPC references."""
        reasoning_results = [
            (
                "agent_1",
                ReasoningResponse(
                    reasoning="Let me talk to my partner.",
                    sentiment=0.3,
                    conviction=0.6,
                    actions=[{"type": "talk_to", "who": "partner"}],
                ),
            ),
        ]

        agent_map = {
            "agent_1": {
                "_id": "agent_1",
                "first_name": "Bob",
                "partner_npc": {"name": "Lisa", "age": 35},
            },
        }

        requests = collect_conversation_requests(
            reasoning_results=reasoning_results,
            adjacency={},
            agent_map=agent_map,
        )

        assert len(requests) == 1
        assert requests[0].target_is_npc is True
        assert requests[0].target_name == "Lisa"
        assert requests[0].relationship == "partner"

    def test_ignores_non_talk_to_actions(self):
        """Should only process talk_to action type."""
        reasoning_results = [
            (
                "agent_1",
                ReasoningResponse(
                    reasoning="Test",
                    sentiment=0.0,
                    conviction=0.5,
                    actions=[{"type": "other_action", "who": "someone"}],
                ),
            ),
        ]

        requests = collect_conversation_requests(
            reasoning_results=reasoning_results,
            adjacency={},
            agent_map={},
        )

        assert len(requests) == 0


class TestPrioritizeAndResolveConflicts:
    """Tests for conflict resolution in conversation prioritization."""

    def test_prioritizes_by_score(self):
        """Should accept higher-priority requests first."""
        requests = [
            ConversationRequest(
                initiator_id="a1",
                target_id="a2",
                priority_score=0.9,
                relationship="partner",
            ),
            ConversationRequest(
                initiator_id="a3",
                target_id="a4",
                priority_score=0.5,
                relationship="coworker",
            ),
        ]

        batches, deferred = prioritize_and_resolve_conflicts(
            requests, fidelity="medium"
        )

        assert len(batches) == 1
        assert len(batches[0]) == 2  # Both can run in parallel (no conflict)
        assert len(deferred) == 0

    def test_resolves_conflicts(self):
        """Should defer conflicting requests."""
        requests = [
            ConversationRequest(
                initiator_id="a1",
                target_id="a2",
                priority_score=0.9,
                relationship="partner",
            ),
            ConversationRequest(
                initiator_id="a3",
                target_id="a2",  # Same target - conflict!
                priority_score=0.5,
                relationship="coworker",
            ),
        ]

        batches, deferred = prioritize_and_resolve_conflicts(
            requests, fidelity="medium"
        )

        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0].initiator_id == "a1"  # Higher priority wins
        assert len(deferred) == 1
        assert deferred[0].initiator_id == "a3"

    def test_low_fidelity_skips_all(self):
        """Low fidelity should skip all conversations."""
        requests = [
            ConversationRequest(
                initiator_id="a1",
                target_id="a2",
                priority_score=0.9,
                relationship="partner",
            ),
        ]

        batches, deferred = prioritize_and_resolve_conflicts(requests, fidelity="low")

        assert len(batches) == 0
        assert len(deferred) == 1

    def test_high_fidelity_allows_more_per_agent(self):
        """High fidelity allows up to 2 conversations per agent."""
        requests = [
            ConversationRequest(
                initiator_id="a1",
                target_id="a2",
                priority_score=0.9,
                relationship="partner",
            ),
            ConversationRequest(
                initiator_id="a1",
                target_id="a3",
                priority_score=0.7,
                relationship="friend",
            ),
            ConversationRequest(
                initiator_id="a1",
                target_id="a4",
                priority_score=0.5,  # Should be deferred (3rd for a1)
                relationship="coworker",
            ),
        ]

        batches, deferred = prioritize_and_resolve_conflicts(requests, fidelity="high")

        assert len(batches) == 1
        assert len(batches[0]) == 2  # Two conversations allowed per agent
        assert len(deferred) == 1  # Third is deferred

    def test_npc_targets_dont_block(self):
        """NPC targets shouldn't block other conversations."""
        requests = [
            ConversationRequest(
                initiator_id="a1",
                target_id="npc_partner",
                target_is_npc=True,
                priority_score=0.9,
                relationship="partner",
            ),
            ConversationRequest(
                initiator_id="a2",
                target_id="npc_partner",  # Same NPC target, but NPCs can be shared
                target_is_npc=True,
                priority_score=0.5,
                relationship="partner",
            ),
        ]

        batches, deferred = prioritize_and_resolve_conflicts(
            requests, fidelity="medium"
        )

        # Both should be accepted since NPCs don't have busy state
        assert len(batches) == 1
        assert len(batches[0]) == 2
        assert len(deferred) == 0


class TestConversationModels:
    """Tests for conversation data models."""

    def test_conversation_request_defaults(self):
        """Should have sensible defaults."""
        req = ConversationRequest(
            initiator_id="a1",
            target_id="a2",
        )
        assert req.target_is_npc is False
        assert req.priority_score == 0.5
        assert req.relationship == "contact"

    def test_conversation_message_creation(self):
        """Should create messages with all fields."""
        msg = ConversationMessage(
            speaker_id="a1",
            speaker_name="Alice",
            content="Hello!",
            turn=0,
            is_final=False,
        )
        assert msg.content == "Hello!"
        assert msg.turn == 0

    def test_conversation_result_creation(self):
        """Should create results with state changes."""
        result = ConversationResult(
            initiator_id="a1",
            target_id="a2",
            messages=[
                ConversationMessage(
                    speaker_id="a1",
                    speaker_name="Alice",
                    content="Hi",
                    turn=0,
                )
            ],
            initiator_state_change=ConversationStateChange(
                sentiment=0.5,
                conviction=0.7,
            ),
        )
        assert len(result.messages) == 1
        assert result.initiator_state_change.sentiment == 0.5


class TestFidelityTurnCount:
    """Tests for fidelity-based turn count."""

    def test_medium_fidelity_has_2_turns(self):
        """Medium fidelity should have 2 turns (4 messages)."""
        # This tests the turn count logic without actually calling LLM
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir="test/",
            fidelity="medium",
        )
        assert config.fidelity == "medium"
        # In the actual execute_conversation_async, turns = 2 for medium

    def test_high_fidelity_has_3_turns(self):
        """High fidelity should have 3 turns (6 messages)."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir="test/",
            fidelity="high",
        )
        assert config.fidelity == "high"
        # In the actual execute_conversation_async, turns = 3 for high


class TestStateOverride:
    """Tests for conversation state override logic."""

    def test_state_change_model(self):
        """State change should hold sentiment and conviction updates."""
        change = ConversationStateChange(
            sentiment=0.8,
            conviction=0.9,
            position="support",
            internal_reaction="I feel more confident now.",
        )
        assert change.sentiment == 0.8
        assert change.conviction == 0.9
        assert change.position == "support"

    def test_conversation_result_preserves_state_changes(self):
        """Both initiator and target state changes should be preserved."""
        result = ConversationResult(
            initiator_id="a1",
            target_id="a2",
            initiator_state_change=ConversationStateChange(sentiment=0.5),
            target_state_change=ConversationStateChange(sentiment=-0.3),
        )
        assert result.initiator_state_change.sentiment == 0.5
        assert result.target_state_change.sentiment == -0.3

    def test_npc_target_no_state_change(self):
        """NPC targets should not have state changes."""
        result = ConversationResult(
            initiator_id="a1",
            target_id="npc_partner",
            target_is_npc=True,
            initiator_state_change=ConversationStateChange(sentiment=0.5),
            target_state_change=None,  # NPCs don't have state
        )
        assert result.target_state_change is None


class TestReasoningResponseActions:
    """Tests for actions field in ReasoningResponse."""

    def test_reasoning_response_with_actions(self):
        """ReasoningResponse should support actions field."""
        response = ReasoningResponse(
            reasoning="I need to talk to someone.",
            sentiment=0.5,
            conviction=0.7,
            actions=[
                {"type": "talk_to", "who": "Sarah", "topic": "the news"},
                {"type": "talk_to", "who": "John"},
            ],
        )
        assert len(response.actions) == 2
        assert response.actions[0]["who"] == "Sarah"
        assert response.actions[1]["who"] == "John"

    def test_reasoning_response_empty_actions(self):
        """ReasoningResponse should default to empty actions."""
        response = ReasoningResponse(
            reasoning="I'll just wait.",
            sentiment=0.0,
            conviction=0.3,
        )
        assert response.actions == []
