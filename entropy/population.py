"""Population creation pipeline for Entropy."""

import random
import uuid
from typing import Any, Callable

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from .llm import chat_completion
from .models import (
    Agent,
    AgentState,
    Cognitive,
    Connection,
    Demographics,
    InformationEnvironment,
    Network,
    ParsedContext,
    Population,
    Psychographics,
    ResearchData,
)
from .search import conduct_research


# =============================================================================
# Step 1: Context Parsing
# =============================================================================


class ParsedContextResponse(BaseModel):
    """Response model for context parsing."""

    size: int = Field(description="Number of agents to create")
    base_population: str = Field(description="Base population (e.g., 'US adults', 'consumers')")
    context_type: str = Field(description="Type: 'subscription', 'ownership', 'membership', 'general'")
    context_entity: str | None = Field(description="Entity like 'Netflix', 'Tesla', or None")
    geography: str | None = Field(description="Geographic scope like 'US', 'California', or None")
    filters: list[str] = Field(default_factory=list, description="Additional filters mentioned")


def parse_context(context: str, default_size: int = 1000) -> ParsedContext:
    """
    Parse natural language context into structured format.

    Example: "2000 Netflix subscribers in the US" ->
        ParsedContext(size=2000, base_population="consumers", context_type="subscription",
                     context_entity="Netflix", geography="US", filters=[])
    """
    response = chat_completion(
        messages=[
            {
                "role": "system",
                "content": f"""Parse the population description into structured components.

Extract:
- size: Number of agents (default {default_size} if not specified)
- base_population: The underlying population (e.g., "US adults", "consumers")
- context_type: One of "subscription", "ownership", "membership", or "general"
- context_entity: The company/product if mentioned (e.g., "Netflix", "Tesla")
- geography: Geographic scope if mentioned
- filters: Any additional filters

Be precise and extract exactly what's stated.""",
            },
            {"role": "user", "content": context},
        ],
        response_format=ParsedContextResponse,
    )

    return ParsedContext(
        size=response.size,
        base_population=response.base_population,
        context_type=response.context_type,
        context_entity=response.context_entity,
        geography=response.geography,
        filters=response.filters,
    )


# =============================================================================
# Step 2: Research (delegated to search.py)
# =============================================================================


# =============================================================================
# Step 3: Distribution Building
# =============================================================================


def _parse_age_range(age_str: str) -> tuple[int, int]:
    """Parse age range string like '25-34' or '65+' into (min, max)."""
    age_str = age_str.strip()
    if "+" in age_str:
        min_age = int(age_str.replace("+", ""))
        return (min_age, min_age + 15)  # Assume 15 year range for 65+
    if "-" in age_str:
        parts = age_str.split("-")
        return (int(parts[0]), int(parts[1]))
    # Single number
    try:
        age = int(age_str)
        return (age, age)
    except ValueError:
        return (30, 40)  # Default


def _parse_income_range(income_str: str) -> tuple[int, int]:
    """Parse income range string like '<30k', '50k-75k', '100k+' into (min, max)."""
    income_str = income_str.lower().strip().replace("$", "").replace(",", "")

    # Handle 'k' notation
    multiplier = 1000 if "k" in income_str else 1
    income_str = income_str.replace("k", "")

    if income_str.startswith("<"):
        return (0, int(income_str[1:]) * multiplier)
    if "+" in income_str:
        min_val = int(income_str.replace("+", "")) * multiplier
        return (min_val, min_val * 2)
    if "-" in income_str:
        parts = income_str.split("-")
        return (int(parts[0]) * multiplier, int(parts[1]) * multiplier)
    try:
        val = int(income_str) * multiplier
        return (val, val)
    except ValueError:
        return (50000, 75000)


class Distributions:
    """Container for sampling distributions built from research."""

    def __init__(self, research: ResearchData, context: ParsedContext):
        self.research = research
        self.context = context
        self._build_distributions()

    def _build_distributions(self):
        """Build scipy distributions from research data."""
        demo = self.research.demographics

        # Age distribution
        self.age_ranges: list[tuple[int, int]] = []
        self.age_weights: list[float] = []
        for age_str, weight in demo.get("age_distribution", {}).items():
            self.age_ranges.append(_parse_age_range(age_str))
            self.age_weights.append(weight)

        if not self.age_weights:
            # Default age distribution
            self.age_ranges = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 80)]
            self.age_weights = [0.12, 0.22, 0.20, 0.18, 0.15, 0.13]

        # Normalize weights
        total = sum(self.age_weights)
        self.age_weights = [w / total for w in self.age_weights]

        # Gender distribution
        self.gender_options = list(demo.get("gender_distribution", {"male": 0.49, "female": 0.51}).keys())
        self.gender_weights = list(demo.get("gender_distribution", {"male": 0.49, "female": 0.51}).values())

        # Income distribution
        self.income_ranges: list[tuple[int, int]] = []
        self.income_weights: list[float] = []
        for income_str, weight in demo.get("income_distribution", {}).items():
            self.income_ranges.append(_parse_income_range(income_str))
            self.income_weights.append(weight)

        if not self.income_weights:
            self.income_ranges = [(0, 30000), (30000, 50000), (50000, 75000), (75000, 100000), (100000, 200000)]
            self.income_weights = [0.20, 0.25, 0.25, 0.15, 0.15]

        total = sum(self.income_weights)
        self.income_weights = [w / total for w in self.income_weights]

        # Education distribution
        self.education_options = list(
            demo.get(
                "education_distribution",
                {"high_school": 0.30, "some_college": 0.20, "bachelors": 0.30, "masters": 0.15, "doctorate": 0.05},
            ).keys()
        )
        self.education_weights = list(
            demo.get(
                "education_distribution",
                {"high_school": 0.30, "some_college": 0.20, "bachelors": 0.30, "masters": 0.15, "doctorate": 0.05},
            ).values()
        )

        # Location distribution
        self.location_options = list(demo.get("location_distribution", {"urban": 0.55, "suburban": 0.35, "rural": 0.10}).keys())
        self.location_weights = list(demo.get("location_distribution", {"urban": 0.55, "suburban": 0.35, "rural": 0.10}).values())

        # Ethnicity distribution
        self.ethnicity_options = list(
            demo.get(
                "ethnicity_distribution",
                {"white": 0.60, "hispanic": 0.18, "black": 0.13, "asian": 0.06, "other": 0.03},
            ).keys()
        )
        self.ethnicity_weights = list(
            demo.get(
                "ethnicity_distribution",
                {"white": 0.60, "hispanic": 0.18, "black": 0.13, "asian": 0.06, "other": 0.03},
            ).values()
        )

        # Marital status (standard distribution)
        self.marital_options = ["single", "married", "divorced", "widowed"]
        self.marital_weights = [0.35, 0.45, 0.15, 0.05]

        # Household size distribution
        self.household_sizes = [1, 2, 3, 4, 5]
        self.household_weights = [0.28, 0.34, 0.15, 0.13, 0.10]

        # Situation distributions (from research)
        self.situation_distributions = self.research.situation_distributions
        self.situation_schema = self.research.situation_schema

    def sample_age(self) -> int:
        """Sample an age value."""
        idx = np.random.choice(len(self.age_ranges), p=self.age_weights)
        min_age, max_age = self.age_ranges[idx]
        return int(np.random.uniform(min_age, max_age))

    def sample_gender(self) -> str:
        """Sample a gender."""
        return np.random.choice(self.gender_options, p=self.gender_weights)

    def sample_income(self, age: int, education: str) -> int:
        """Sample income with correlation to age and education."""
        # Base income from distribution
        idx = np.random.choice(len(self.income_ranges), p=self.income_weights)
        min_inc, max_inc = self.income_ranges[idx]
        base_income = int(np.random.uniform(min_inc, max_inc))

        # Age modifier (peak at 45-55)
        age_modifier = 1.0
        if age < 25:
            age_modifier = 0.6
        elif age < 35:
            age_modifier = 0.85
        elif age < 55:
            age_modifier = 1.1
        elif age > 65:
            age_modifier = 0.8

        # Education modifier
        edu_modifiers = {
            "high_school": 0.7,
            "some_college": 0.85,
            "bachelors": 1.0,
            "masters": 1.2,
            "doctorate": 1.4,
        }
        edu_modifier = edu_modifiers.get(education, 1.0)

        # Apply modifiers with some noise
        modified = base_income * age_modifier * edu_modifier
        noise = np.random.normal(1.0, 0.1)
        return max(15000, int(modified * noise))

    def sample_education(self) -> str:
        """Sample education level."""
        return np.random.choice(self.education_options, p=self.education_weights)

    def sample_location(self) -> dict:
        """Sample location."""
        urban_rural = np.random.choice(self.location_options, p=self.location_weights)
        # US states (simplified)
        states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA"]
        state_weights = [0.12, 0.09, 0.07, 0.06, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02]
        # Normalize and pad
        remaining = 1 - sum(state_weights)
        state_weights.append(remaining)
        states.append("Other")

        state = np.random.choice(states, p=state_weights)
        return {"state": state, "urban_rural": urban_rural}

    def sample_ethnicity(self) -> str:
        """Sample ethnicity."""
        return np.random.choice(self.ethnicity_options, p=self.ethnicity_weights)

    def sample_marital_status(self, age: int) -> str:
        """Sample marital status with age correlation."""
        weights = self.marital_weights.copy()

        # Young people more likely single
        if age < 25:
            weights = [0.75, 0.20, 0.04, 0.01]
        elif age < 35:
            weights = [0.45, 0.45, 0.08, 0.02]
        elif age > 65:
            weights = [0.15, 0.50, 0.15, 0.20]

        return np.random.choice(self.marital_options, p=weights)

    def sample_household_size(self, marital_status: str) -> int:
        """Sample household size with marital status correlation."""
        weights = self.household_weights.copy()

        if marital_status == "single":
            weights = [0.55, 0.25, 0.10, 0.07, 0.03]
        elif marital_status == "married":
            weights = [0.05, 0.35, 0.25, 0.20, 0.15]

        return int(np.random.choice(self.household_sizes, p=weights))

    def sample_situation(self) -> dict[str, Any]:
        """Sample situation attributes based on dynamically researched schema."""
        situation = {}

        for schema in self.situation_schema:
            dist = self.situation_distributions.get(schema.name, {})

            if schema.field_type == "int":
                min_val = dist.get("min", schema.min_value or 0)
                max_val = dist.get("max", schema.max_value or 100)
                mean = dist.get("mean", (min_val + max_val) / 2)
                std = dist.get("std", (max_val - min_val) / 4)
                value = int(np.clip(np.random.normal(mean, std), min_val, max_val))
                situation[schema.name] = value

            elif schema.field_type == "float":
                min_val = dist.get("min", schema.min_value or 0.0)
                max_val = dist.get("max", schema.max_value or 1.0)
                mean = dist.get("mean", (min_val + max_val) / 2)
                std = dist.get("std", (max_val - min_val) / 4)
                value = float(np.clip(np.random.normal(mean, std), min_val, max_val))
                situation[schema.name] = round(value, 2)

            elif schema.field_type == "str":
                options = dist.get("options") or schema.options or ["unknown"]
                weights = dist.get("weights")
                if weights and len(weights) == len(options):
                    situation[schema.name] = np.random.choice(options, p=weights)
                else:
                    situation[schema.name] = np.random.choice(options)

            elif schema.field_type == "list":
                options = dist.get("options") or schema.options or []
                if options:
                    k = np.random.randint(0, min(3, len(options)) + 1)
                    situation[schema.name] = list(np.random.choice(options, size=k, replace=False))
                else:
                    situation[schema.name] = []

        return situation


# =============================================================================
# Step 4: Agent Sampling
# =============================================================================

# Common values and interests by personality
VALUES_BY_TRAIT = {
    "high_openness": ["creativity", "curiosity", "adventure", "diversity", "independence"],
    "low_openness": ["tradition", "stability", "security", "practicality", "conformity"],
    "high_conscientiousness": ["achievement", "order", "duty", "self-discipline", "reliability"],
    "high_extraversion": ["excitement", "social_recognition", "leadership", "fun", "connection"],
    "high_agreeableness": ["harmony", "compassion", "cooperation", "trust", "kindness"],
    "high_neuroticism": ["security", "safety", "support", "reassurance", "stability"],
}

INTERESTS_POOL = [
    "technology",
    "sports",
    "music",
    "movies",
    "gaming",
    "cooking",
    "travel",
    "fitness",
    "reading",
    "art",
    "nature",
    "politics",
    "finance",
    "fashion",
    "photography",
    "gardening",
    "crafts",
    "cars",
    "food",
    "pets",
]

NEWS_SOURCES = ["CNN", "Fox News", "MSNBC", "NYT", "WSJ", "Local News", "NPR", "BBC", "Online Only", "None"]
SOCIAL_MEDIA = ["Facebook", "Instagram", "Twitter/X", "TikTok", "YouTube", "LinkedIn", "Reddit", "None"]
OCCUPATIONS = [
    "Software Engineer",
    "Teacher",
    "Nurse",
    "Sales Representative",
    "Manager",
    "Accountant",
    "Administrative Assistant",
    "Customer Service",
    "Retail Worker",
    "Healthcare Worker",
    "Construction Worker",
    "Driver",
    "Marketing Professional",
    "Engineer",
    "Consultant",
    "Freelancer",
    "Student",
    "Retired",
    "Unemployed",
    "Business Owner",
]


def sample_psychographics(age: int, education: str) -> Psychographics:
    """Sample psychographic traits with some correlations."""
    # Big Five with some correlation to demographics
    openness = np.clip(np.random.beta(2, 2), 0, 1)
    conscientiousness = np.clip(np.random.beta(2.5, 2), 0, 1)
    extraversion = np.clip(np.random.beta(2, 2), 0, 1)
    agreeableness = np.clip(np.random.beta(2.5, 2), 0, 1)
    neuroticism = np.clip(np.random.beta(2, 2.5), 0, 1)

    # Age correlations
    if age > 50:
        conscientiousness = min(1, conscientiousness + 0.1)
        openness = max(0, openness - 0.05)
    if age < 30:
        openness = min(1, openness + 0.1)

    # Education correlations
    if education in ["masters", "doctorate"]:
        openness = min(1, openness + 0.1)

    # Sample values based on dominant traits
    values = []
    if openness > 0.6:
        values.extend(random.sample(VALUES_BY_TRAIT["high_openness"], 2))
    else:
        values.extend(random.sample(VALUES_BY_TRAIT["low_openness"], 2))
    if conscientiousness > 0.6:
        values.extend(random.sample(VALUES_BY_TRAIT["high_conscientiousness"], 1))
    if extraversion > 0.6:
        values.extend(random.sample(VALUES_BY_TRAIT["high_extraversion"], 1))

    values = list(set(values))[:5]

    # Sample interests
    num_interests = np.random.randint(3, 7)
    interests = random.sample(INTERESTS_POOL, num_interests)

    return Psychographics(
        openness=round(openness, 2),
        conscientiousness=round(conscientiousness, 2),
        extraversion=round(extraversion, 2),
        agreeableness=round(agreeableness, 2),
        neuroticism=round(neuroticism, 2),
        values=values,
        interests=interests,
    )


def sample_cognitive(psychographics: Psychographics, education: str) -> Cognitive:
    """Sample cognitive traits with correlations to personality and education."""
    # Information processing style
    if psychographics.openness > 0.6 and psychographics.conscientiousness > 0.5:
        processing = "analytical"
    elif psychographics.openness < 0.4:
        processing = "intuitive"
    else:
        processing = np.random.choice(["analytical", "intuitive", "balanced"], p=[0.3, 0.3, 0.4])

    # Openness to change correlates with Big Five openness
    openness_to_change = np.clip(psychographics.openness + np.random.normal(0, 0.1), 0, 1)

    # Trust in institutions
    base_trust = 0.5
    if education in ["masters", "doctorate"]:
        base_trust += 0.1
    if psychographics.agreeableness > 0.6:
        base_trust += 0.1
    trust_in_institutions = np.clip(base_trust + np.random.normal(0, 0.15), 0, 1)

    # Confirmation bias (higher neuroticism and lower openness = more bias)
    confirmation_bias = np.clip(
        0.5 + (psychographics.neuroticism - 0.5) * 0.3 - (psychographics.openness - 0.5) * 0.2 + np.random.normal(0, 0.1),
        0,
        1,
    )

    # Persuadability (higher agreeableness, lower conscientiousness = more persuadable)
    persuadability = np.clip(
        0.5
        + (psychographics.agreeableness - 0.5) * 0.3
        - (psychographics.conscientiousness - 0.5) * 0.2
        + np.random.normal(0, 0.1),
        0,
        1,
    )

    return Cognitive(
        information_processing=processing,
        openness_to_change=round(openness_to_change, 2),
        trust_in_institutions=round(trust_in_institutions, 2),
        confirmation_bias=round(confirmation_bias, 2),
        persuadability=round(persuadability, 2),
    )


def sample_information_env(age: int, psychographics: Psychographics) -> InformationEnvironment:
    """Sample information environment based on demographics and personality."""
    # News sources (age and personality dependent)
    num_sources = np.random.randint(1, 4)
    if age > 50:
        news_weights = [0.15, 0.15, 0.10, 0.10, 0.10, 0.20, 0.10, 0.05, 0.03, 0.02]
    else:
        news_weights = [0.10, 0.05, 0.05, 0.10, 0.05, 0.10, 0.05, 0.05, 0.30, 0.15]
    news_sources = list(np.random.choice(NEWS_SOURCES, size=num_sources, replace=False, p=news_weights))

    # Social media (age dependent)
    num_social = np.random.randint(1, 5)
    if age < 30:
        social_weights = [0.10, 0.20, 0.10, 0.25, 0.20, 0.05, 0.08, 0.02]
    elif age < 50:
        social_weights = [0.20, 0.20, 0.10, 0.10, 0.15, 0.10, 0.10, 0.05]
    else:
        social_weights = [0.35, 0.10, 0.05, 0.02, 0.15, 0.05, 0.03, 0.25]
    social_media = list(np.random.choice(SOCIAL_MEDIA, size=num_social, replace=False, p=social_weights))

    # Media hours (extraversion and age dependent)
    base_hours = np.random.uniform(1, 6)
    if psychographics.extraversion < 0.4:
        base_hours += 1
    if age < 30:
        base_hours += 1
    media_hours = round(min(base_hours, 10), 1)

    # Trust in media
    trust_in_media = np.clip(np.random.beta(2, 2.5), 0, 1)

    # Exposure rate (extraversion and openness dependent)
    exposure_rate = np.clip(
        0.5 + (psychographics.extraversion - 0.5) * 0.2 + (psychographics.openness - 0.5) * 0.2 + np.random.normal(0, 0.1),
        0,
        1,
    )

    return InformationEnvironment(
        news_sources=news_sources,
        social_media=social_media,
        media_hours_daily=media_hours,
        trust_in_media=round(trust_in_media, 2),
        exposure_rate=round(exposure_rate, 2),
    )


def sample_agent(distributions: Distributions) -> Agent:
    """Sample a single agent from distributions."""
    agent_id = str(uuid.uuid4())[:8]

    # Demographics (with correlations)
    age = distributions.sample_age()
    gender = distributions.sample_gender()
    education = distributions.sample_education()
    income = distributions.sample_income(age, education)
    location = distributions.sample_location()
    ethnicity = distributions.sample_ethnicity()
    marital_status = distributions.sample_marital_status(age)
    household_size = distributions.sample_household_size(marital_status)

    occupation = np.random.choice(OCCUPATIONS)

    demographics = Demographics(
        age=age,
        gender=gender,
        income=income,
        education=education,
        occupation=occupation,
        location=location,
        ethnicity=ethnicity,
        marital_status=marital_status,
        household_size=household_size,
    )

    # Psychographics (correlated with demographics)
    psychographics = sample_psychographics(age, education)

    # Cognitive (correlated with psychographics and education)
    cognitive = sample_cognitive(psychographics, education)

    # Information environment (correlated with age and psychographics)
    information_env = sample_information_env(age, psychographics)

    # Situation (from dynamically researched schema)
    situation = distributions.sample_situation()

    return Agent(
        id=agent_id,
        demographics=demographics,
        psychographics=psychographics,
        cognitive=cognitive,
        information_env=information_env,
        situation=situation,
        network=Network(),
        persona="",
        state=AgentState(),
    )


def sample_agents(distributions: Distributions, n: int, progress_callback: Callable[[int], None] | None = None) -> list[Agent]:
    """Sample n agents from distributions."""
    agents = []
    for i in range(n):
        agents.append(sample_agent(distributions))
        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1)
    return agents


# =============================================================================
# Step 5: Network Generation
# =============================================================================


def calculate_agent_similarity(agent1: Agent, agent2: Agent) -> float:
    """Calculate similarity score between two agents (0-1)."""
    score = 0.0
    weights_sum = 0.0

    # Age similarity (weight: 2)
    age_diff = abs(agent1.demographics.age - agent2.demographics.age)
    age_sim = max(0, 1 - age_diff / 30)  # 30 year diff = 0 similarity
    score += 2 * age_sim
    weights_sum += 2

    # Location similarity (weight: 3)
    if agent1.demographics.location.get("state") == agent2.demographics.location.get("state"):
        score += 3
    if agent1.demographics.location.get("urban_rural") == agent2.demographics.location.get("urban_rural"):
        score += 1
    weights_sum += 4

    # Education similarity (weight: 1)
    edu_order = ["high_school", "some_college", "bachelors", "masters", "doctorate"]
    try:
        edu_diff = abs(edu_order.index(agent1.demographics.education) - edu_order.index(agent2.demographics.education))
        score += 1 - edu_diff / 4
    except ValueError:
        pass
    weights_sum += 1

    # Income similarity (weight: 2)
    inc_diff = abs(agent1.demographics.income - agent2.demographics.income)
    inc_sim = max(0, 1 - inc_diff / 100000)
    score += 2 * inc_sim
    weights_sum += 2

    # Personality similarity - Big Five (weight: 2)
    p1, p2 = agent1.psychographics, agent2.psychographics
    personality_diff = (
        abs(p1.openness - p2.openness)
        + abs(p1.conscientiousness - p2.conscientiousness)
        + abs(p1.extraversion - p2.extraversion)
        + abs(p1.agreeableness - p2.agreeableness)
        + abs(p1.neuroticism - p2.neuroticism)
    ) / 5
    score += 2 * (1 - personality_diff)
    weights_sum += 2

    # Shared interests (weight: 2)
    shared_interests = len(set(p1.interests) & set(p2.interests))
    max_interests = max(len(p1.interests), len(p2.interests), 1)
    score += 2 * (shared_interests / max_interests)
    weights_sum += 2

    return score / weights_sum


def generate_network(agents: list[Agent], k: int = 6, p: float = 0.1) -> list[Agent]:
    """
    Generate social network using hybrid Watts-Strogatz + similarity weighting.

    Args:
        agents: List of agents
        k: Average number of connections per agent
        p: Rewiring probability for small-world effect
    """
    n = len(agents)
    if n < k + 1:
        k = max(2, n - 1)

    # Create Watts-Strogatz small-world graph
    G = nx.watts_strogatz_graph(n, k, p)

    # Build agent lookup
    agent_by_idx = {i: agent for i, agent in enumerate(agents)}

    # Add edges with similarity-based weights
    for i, agent in enumerate(agents):
        connections = []

        for neighbor_idx in G.neighbors(i):
            neighbor = agent_by_idx[neighbor_idx]
            similarity = calculate_agent_similarity(agent, neighbor)

            # Strength based on similarity (0.3 - 1.0 range)
            strength = 0.3 + 0.7 * similarity

            # Connection type based on shared attributes
            if agent.demographics.location.get("state") == neighbor.demographics.location.get("state"):
                conn_type = "local"
            elif len(set(agent.psychographics.interests) & set(neighbor.psychographics.interests)) >= 2:
                conn_type = "interest"
            else:
                conn_type = "social"

            connections.append(
                Connection(
                    target_id=neighbor.id,
                    strength=round(strength, 2),
                    connection_type=conn_type,
                )
            )

        # Calculate influence score (based on degree and average connection strength)
        degree = len(connections)
        avg_strength = sum(c.strength for c in connections) / max(degree, 1)
        influence = (degree / k) * avg_strength  # Normalized by expected degree

        agent.network = Network(
            connections=connections,
            influence_score=round(min(influence, 2.0), 2),  # Cap at 2.0
        )

    return agents


# =============================================================================
# Step 6: Persona Synthesis
# =============================================================================


def synthesize_persona(agent: Agent, context: ParsedContext) -> str:
    """Generate a natural language persona from agent attributes using templates."""

    # Pronouns
    pronouns = {"male": ("He", "his", "him"), "female": ("She", "her", "her")}
    subj, poss, obj = pronouns.get(agent.demographics.gender, ("They", "their", "them"))

    # Personality descriptions
    def describe_trait(value: float, high_desc: str, low_desc: str) -> str:
        if value > 0.7:
            return f"very {high_desc}"
        elif value > 0.55:
            return f"somewhat {high_desc}"
        elif value < 0.3:
            return f"quite {low_desc}"
        elif value < 0.45:
            return f"somewhat {low_desc}"
        else:
            return f"moderately {high_desc}"

    openness_desc = describe_trait(agent.psychographics.openness, "open to new experiences", "traditional")
    extraversion_desc = describe_trait(agent.psychographics.extraversion, "outgoing", "introverted")
    agreeableness_desc = describe_trait(agent.psychographics.agreeableness, "cooperative", "competitive")
    conscientiousness_desc = describe_trait(agent.psychographics.conscientiousness, "organized", "spontaneous")

    # Information processing
    processing_desc = {
        "analytical": "prefers to analyze information thoroughly before forming opinions",
        "intuitive": "tends to go with gut feelings and first impressions",
        "balanced": "balances analytical thinking with intuition",
    }.get(agent.cognitive.information_processing, "")

    # Trust level
    if agent.cognitive.trust_in_institutions > 0.7:
        trust_desc = "generally trusts major institutions and established sources"
    elif agent.cognitive.trust_in_institutions < 0.3:
        trust_desc = "tends to be skeptical of institutional narratives"
    else:
        trust_desc = "has mixed feelings about institutional trustworthiness"

    # Situation-specific details
    situation_parts = []
    for key, value in agent.situation.items():
        # Format nicely
        key_readable = key.replace("_", " ")
        if isinstance(value, float):
            value = f"{value:.1f}"
        elif isinstance(value, list):
            value = ", ".join(str(v) for v in value) if value else "none"
        situation_parts.append(f"{key_readable}: {value}")

    situation_text = "; ".join(situation_parts) if situation_parts else ""

    # Build persona
    context_entity = context.context_entity or "this service"

    persona = f"""A {agent.demographics.age}-year-old {agent.demographics.gender} from {agent.demographics.location.get('urban_rural', 'urban')} {agent.demographics.location.get('state', 'US')}, working as a {agent.demographics.occupation}. {subj} earns around ${agent.demographics.income:,} annually and has a {agent.demographics.education.replace('_', ' ')} education.

{subj} is {openness_desc}, {extraversion_desc}, and {agreeableness_desc}. {subj} {processing_desc} and {trust_desc}.

{subj} gets news from {', '.join(agent.information_env.news_sources) if agent.information_env.news_sources else 'various sources'} and uses {', '.join(agent.information_env.social_media) if agent.information_env.social_media else 'limited social media'}. {subj} spends about {agent.information_env.media_hours_daily} hours daily consuming media.

{poss.capitalize()} values include {', '.join(agent.psychographics.values[:3])}. {subj} is interested in {', '.join(agent.psychographics.interests[:4])}.

As a {context_entity} customer: {situation_text}."""

    return persona.strip()


def synthesize_personas(agents: list[Agent], context: ParsedContext, progress_callback: Callable[[int], None] | None = None) -> list[Agent]:
    """Generate personas for all agents."""
    for i, agent in enumerate(agents):
        agent.persona = synthesize_persona(agent, context)
        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1)
    return agents


# =============================================================================
# Step 7: Full Pipeline
# =============================================================================


class ProgressCallbacks:
    """Container for progress callbacks."""

    def __init__(self):
        self.on_step: Callable[[str], None] | None = None
        self.on_progress: Callable[[str, int, int], None] | None = None
        self.on_complete: Callable[[str], None] | None = None


def create_population(
    context: str,
    name: str,
    seed: int | None = None,
    callbacks: ProgressCallbacks | None = None,
) -> Population:
    """
    Create a population from natural language context.

    This is the main entry point for Phase 1.

    Args:
        context: Natural language description like "2000 Netflix subscribers in the US"
        name: Name for the population
        seed: Random seed for reproducibility
        callbacks: Optional progress callbacks

    Returns:
        Population object with all agents, network, and personas
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cb = callbacks or ProgressCallbacks()

    # Step 1: Parse context
    if cb.on_step:
        cb.on_step("Parsing context...")
    parsed = parse_context(context)
    if cb.on_complete:
        cb.on_complete("Parsing context")

    # Step 2: Research
    if cb.on_step:
        cb.on_step("Researching population...")
    research = conduct_research(parsed)
    if cb.on_complete:
        cb.on_complete("Researching population")

    # Step 3: Build distributions
    if cb.on_step:
        cb.on_step("Building distributions...")
    distributions = Distributions(research, parsed)
    if cb.on_complete:
        cb.on_complete("Building distributions")

    # Step 4: Sample agents
    if cb.on_step:
        cb.on_step(f"Generating {parsed.size} agents...")

    def agent_progress(n: int):
        if cb.on_progress:
            cb.on_progress("Generating agents", n, parsed.size)

    agents = sample_agents(distributions, parsed.size, progress_callback=agent_progress)
    if cb.on_complete:
        cb.on_complete(f"Generating {parsed.size} agents")

    # Step 5: Generate network
    if cb.on_step:
        cb.on_step("Building social network...")
    agents = generate_network(agents)
    if cb.on_complete:
        cb.on_complete("Building social network")

    # Step 6: Synthesize personas
    if cb.on_step:
        cb.on_step("Synthesizing personas...")

    def persona_progress(n: int):
        if cb.on_progress:
            cb.on_progress("Synthesizing personas", n, parsed.size)

    agents = synthesize_personas(agents, parsed, progress_callback=persona_progress)
    if cb.on_complete:
        cb.on_complete("Synthesizing personas")

    # Create population
    population = Population(
        name=name,
        size=parsed.size,
        context_raw=context,
        context_parsed=parsed,
        research=research,
        agents=agents,
    )

    return population

