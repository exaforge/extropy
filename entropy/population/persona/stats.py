"""Population statistics computation for relative positioning.

Computes mean, std, min, max for each numeric attribute from sampled agents.
Used to calculate z-scores for relative positioning in personas.
"""

from typing import Any

from .config import PopulationStats


def compute_population_stats(
    agents: list[dict[str, Any]],
    numeric_attributes: list[str] | None = None,
) -> PopulationStats:
    """Compute population statistics from sampled agents.
    
    Args:
        agents: List of sampled agent dictionaries
        numeric_attributes: Optional list of attribute names to compute stats for.
                          If None, computes for all numeric attributes found.
    
    Returns:
        PopulationStats with mean, std, min, max for each attribute
    """
    if not agents:
        return PopulationStats()
    
    # Collect all numeric values per attribute
    attr_values: dict[str, list[float]] = {}
    
    for agent in agents:
        for key, value in agent.items():
            if key.startswith("_"):
                continue
            if numeric_attributes is not None and key not in numeric_attributes:
                continue
            
            # Try to convert to float
            if isinstance(value, bool):
                # Treat booleans as 0/1 for stats
                attr_values.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                attr_values.setdefault(key, []).append(float(value))
            elif isinstance(value, str):
                # Try to parse as number
                try:
                    attr_values.setdefault(key, []).append(float(value))
                except ValueError:
                    pass  # Not numeric, skip
    
    # Compute stats for each attribute
    stats: dict[str, dict[str, float]] = {}
    
    for attr_name, values in attr_values.items():
        if not values:
            continue
        
        n = len(values)
        mean = sum(values) / n
        
        # Compute std
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = variance ** 0.5
        else:
            std = 0.0
        
        stats[attr_name] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "count": float(n),
        }
    
    return PopulationStats(stats=stats)
