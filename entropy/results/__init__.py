"""Results module for loading and displaying simulation results.

Provides utilities for reading simulation results from the output
directory and formatting them for display.

Usage:
    from entropy.results import load_results, ResultsReader

    # Load results
    reader = load_results("results/my_simulation/")

    # Get summary
    summary = reader.get_summary()

    # Get segment breakdown
    segments = reader.compute_segment("age_bracket")

    # Get timeline
    timeline = reader.get_timeline()

    # Get single agent
    agent = reader.get_agent_state("agent_001")
"""

from .models import (
    SimulationSummary,
    AgentFinalState,
    SegmentAggregate,
    TimelinePoint,
    RunMeta,
    SimulationResults,
)

from .reader import (
    ResultsReader,
    load_results,
)

from .formatter import (
    display_summary,
    display_segment_breakdown,
    display_timeline,
    display_agent,
    display_quick_stats,
    display_help,
    format_percentage,
    format_sentiment,
    format_duration,
    make_bar,
)

__all__ = [
    # Models
    "SimulationSummary",
    "AgentFinalState",
    "SegmentAggregate",
    "TimelinePoint",
    "RunMeta",
    "SimulationResults",
    # Reader
    "ResultsReader",
    "load_results",
    # Formatter
    "display_summary",
    "display_segment_breakdown",
    "display_timeline",
    "display_agent",
    "display_quick_stats",
    "display_help",
    "format_percentage",
    "format_sentiment",
    "format_duration",
    "make_bar",
]
