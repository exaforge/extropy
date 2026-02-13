"""Graph utilities for dependency analysis.

This module provides graph algorithms used across the codebase,
particularly for dependency validation and ordering.
"""

from collections import defaultdict


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected.

    Attributes:
        cycle: List of node names forming the cycle
    """

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        cycle_str = " → ".join(cycle + [cycle[0]])
        super().__init__(f"Circular dependency: {cycle_str}")


def topological_sort(dependencies: dict[str, list[str]]) -> list[str]:
    """Topological sort of a dependency graph using Kahn's algorithm.

    Args:
        dependencies: Mapping of node_name -> list of dependency names.
                     Dependencies are nodes that must come BEFORE this node.

    Returns:
        List of node names in valid execution order (dependencies first).

    Raises:
        CircularDependencyError: If the graph contains a cycle.

    Example:
        >>> deps = {"a": [], "b": ["a"], "c": ["a", "b"]}
        >>> topological_sort(deps)
        ['a', 'b', 'c']
    """
    # Build adjacency list and in-degree count
    # graph[x] = list of nodes that depend on x
    graph: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {name: 0 for name in dependencies}
    all_nodes = set(dependencies.keys())

    for node, deps in dependencies.items():
        for dep in deps:
            # Only count dependencies on nodes we're tracking
            if dep in all_nodes:
                graph[dep].append(node)
                in_degree[node] += 1

    # Start with nodes that have no dependencies
    queue = sorted([name for name, degree in in_degree.items() if degree == 0])
    order: list[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)

        # Reduce in-degree for dependents
        for dependent in sorted(graph[node]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
                queue.sort()  # Maintain deterministic order

    # Check for cycles
    if len(order) != len(dependencies):
        # Find the cycle for error reporting
        remaining = {name for name in dependencies if name not in order}
        cycle = _find_cycle(dependencies, remaining)
        raise CircularDependencyError(cycle)

    return order


def _find_cycle(
    dependencies: dict[str, list[str]],
    candidates: set[str],
) -> list[str]:
    """Find a cycle in the dependency graph among candidate nodes."""
    # Use DFS to find a cycle
    visited: set[str] = set()
    path: list[str] = []
    path_set: set[str] = set()

    def dfs(node: str) -> list[str] | None:
        if node in path_set:
            # Found cycle - extract it
            cycle_start = path.index(node)
            return path[cycle_start:]
        if node in visited or node not in candidates:
            return None

        visited.add(node)
        path.append(node)
        path_set.add(node)

        for dep in dependencies.get(node, []):
            if dep in candidates:
                result = dfs(dep)
                if result:
                    return result

        path.pop()
        path_set.remove(node)
        return None

    for start in sorted(candidates):
        result = dfs(start)
        if result:
            return result

    # Fallback if cycle not found (shouldn't happen)
    return list(candidates)[:3]
