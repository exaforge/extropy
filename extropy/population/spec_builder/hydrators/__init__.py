"""Hydrators package for split attribute hydration.

Each hydration strategy has its own module:
- independent.py: Research distributions for independent attributes (Step 2a)
- derived.py: Specify formulas for derived attributes (Step 2b)
- conditional.py: Research base distributions and modifiers (Steps 2c + 2d)
- household.py: Research household composition parameters (Step 2e)
- name.py: Research culturally-appropriate names (Step 2f)
"""

from .independent import hydrate_independent
from .derived import hydrate_derived
from .conditional import hydrate_conditional_base, hydrate_conditional_modifiers
from .household import hydrate_household_config
from .name import hydrate_name_config

__all__ = [
    "hydrate_independent",
    "hydrate_derived",
    "hydrate_conditional_base",
    "hydrate_conditional_modifiers",
    "hydrate_household_config",
    "hydrate_name_config",
]
