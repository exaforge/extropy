"""Hydrators package for split attribute hydration.

Each hydration strategy has its own module:
- independent.py: Research distributions for independent attributes (Step 2a)
- derived.py: Specify formulas for derived attributes (Step 2b)
- conditional.py: Research base distributions and modifiers (Steps 2c + 2d)
"""

from .independent import hydrate_independent
from .derived import hydrate_derived
from .conditional import hydrate_conditional_base, hydrate_conditional_modifiers

__all__ = [
    "hydrate_independent",
    "hydrate_derived",
    "hydrate_conditional_base",
    "hydrate_conditional_modifiers",
]
