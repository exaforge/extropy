#!/usr/bin/env python3
"""Quick test to verify the LLM provider abstraction works.

Usage:
    # Test with default config
    uv run python scripts/test_provider.py

    # Test with Claude pipeline
    uv run python scripts/test_provider.py --provider claude
"""

import sys

from entropy.config import get_config
from entropy.core.llm import simple_call, agentic_research


def main() -> int:
    config = get_config()
    provider = config.pipeline.provider
    print("=" * 60)
    print(f"Testing pipeline provider: {provider.upper()}")
    print("=" * 60)

    # Test 1: Simple call
    print("\n[Test 1] simple_call...")
    try:
        result = simple_call(
            prompt="What is 2 + 2? Return just the number.",
            response_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "integer"},
                },
                "required": ["answer"],
            },
            schema_name="math_test",
            log=False,
        )
        print(f"  Result: {result}")
        assert result.get("answer") == 4, f"Expected 4, got {result.get('answer')}"
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    # Test 2: Agentic research (web search)
    print("\n[Test 2] agentic_research (web search)...")
    try:
        result, sources = agentic_research(
            prompt="What is the current population of Germany? Return the approximate number in millions.",
            response_schema={
                "type": "object",
                "properties": {
                    "population_millions": {"type": "number"},
                    "source": {"type": "string"},
                },
                "required": ["population_millions"],
            },
            schema_name="population_test",
            log=False,
        )
        print(f"  Result: {result}")
        print(f"  Sources found: {len(sources)}")

        pop = result.get("population_millions", 0)
        assert 70 < pop < 100, f"Expected ~83, got {pop}"
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
