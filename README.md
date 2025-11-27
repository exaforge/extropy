# Entropy

Simulate how populations respond to scenarios. Create synthetic populations grounded in real-world data, simulate how they react to events, and watch opinions evolve through social networks over time.

## Installation

```bash
# Clone and install
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Usage

```bash
# Create a population
entropy create "2000 Netflix subscribers in the US" --name netflix_us

# List populations
entropy list

# Inspect a population
entropy inspect netflix_us

# Delete a population
entropy delete netflix_us
```

## Three Phases

| Phase                            | What It Does                                          | LLM               |
| -------------------------------- | ----------------------------------------------------- | ----------------- |
| **Phase 1: Population Creation** | Create N agents from natural language context         | OpenAI API        |
| **Phase 2: Scenario Injection**  | Define what happens to this population                | OpenAI API        |
| **Phase 3: Simulation**          | Agents respond; opinions evolve with social influence | LM Studio (local) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```
