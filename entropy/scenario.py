"""Phase 2: Scenario injection for Entropy.

This module will handle creating and managing scenarios that can be
injected into populations for simulation.

TODO: Implement in Phase 2
- parse_scenario() - Parse natural language scenario description
- research_scenario() - Research real-world context for the scenario
- create_scenario() - Create scenario with event details and channels
- load_scenario() / save_scenario() - Database operations
"""

from .models import Scenario


def create_scenario(description: str, population_name: str, name: str) -> Scenario:
    """
    Create a scenario from natural language description.

    This will be implemented in Phase 2.
    """
    raise NotImplementedError("Phase 2 not yet implemented")

