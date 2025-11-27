"""Test script for agentic search with GPT-5.

Pipeline:
1. GPT-5 agentic search (web search + reasoning + structured output)
2. GPT-5-mini validation (filters aggregate stats, keeps per-agent only)
"""

import time

from entropy.models import ParsedContext
from entropy.search import conduct_research

# Create a test context (what parse_context() would output)
context = ParsedContext(
    size=100,
    base_population="consumers",
    context_type="subscription",
    context_entity="Netflix",
    geography="US",
    filters=[],
)

print("=" * 60)
print(f"Researching: {context.context_entity} {context.context_type}s in {context.geography}")
print("Step 1: GPT-5 agentic search (demographics + situation)")
print("Step 2: GPT-5-instant validation (filter aggregate stats)")
print("=" * 60)
print()

# Run the research - ONE call does everything
start_time = time.time()
research = conduct_research(context, reasoning_effort="medium")
elapsed = time.time() - start_time

# Display results
print("DEMOGRAPHICS:")
print("-" * 40)
for key, value in research.demographics.items():
    if value:  # Only show non-empty
        print(f"  {key}:")
        for k, v in value.items():
            pct = f"{v:.1%}" if isinstance(v, float) else str(v)
            print(f"    {k}: {pct}")

print()
print("SITUATION SCHEMA (per-agent only, aggregates filtered):")
print("-" * 40)
for attr in research.situation_schema:
    print(f"  • {attr.name} ({attr.field_type})")
    print(f"    {attr.description}")
    if attr.min_value is not None:
        print(f"    Range: {attr.min_value} - {attr.max_value}")
    if attr.options:
        print(f"    Options: {attr.options}")

print()
print("GROUNDING:")
print("-" * 40)
print(f"  Level: {research.grounding_level}")
print(f"  Sources: {len(research.sources)}")
for url in research.sources[:5]:
    truncated = url[:60] + "..." if len(url) > 60 else url
    print(f"    • {truncated}")

print()
print("=" * 60)
print(f"Done in {elapsed:.1f}s")
