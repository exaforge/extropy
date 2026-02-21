# Technical Instructions: Simulating Congestion Tax Response with Extropy

**Prepared for:** Austin Transportation Department – Urban Planning & Data Team
**Deliverable:** `mode_shift_report.json` containing simulated commuter response data
**Estimated Time:** 20–30 minutes

---

## Prerequisites

Before beginning, ensure your workstation meets the following requirements:

| Requirement | Specification |
|-------------|---------------|
| Operating System | macOS 12+, Ubuntu 20.04+, or Windows 11 with WSL2 |
| Python | 3.10 or higher |
| RAM | 8GB minimum (16GB recommended for 500+ agents) |
| API Key | Anthropic or OpenAI API key with active billing |

To verify your Python version, open a terminal and run:

```bash
python3 --version
```

If the output shows `Python 3.10.x` or higher, proceed to Step 1.

---

## Step 1: Install the Extropy CLI

Install the Extropy command-line interface using pip:

```bash
pip install extropy-cli
```

Wait for the installation to complete. You should see output ending with:

```
Successfully installed extropy-cli-x.x.x
```

To verify the installation, run:

```bash
extropy --version
```

---

## Step 2: Configure Your API Key

Extropy requires an LLM provider to generate agent populations. Set your API key as an environment variable:

**For Anthropic (recommended):**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**For OpenAI:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

To make this permanent, add the export line to your `~/.bashrc` or `~/.zshrc` file.

---

## Step 3: Initialize the Study Population

Create a new study folder by running the `extropy spec` command with a description of your target population:

```bash
extropy spec "Adult commuters in Austin, Texas who drive to work in the downtown core. Include variation in income levels, commute distances, access to public transit, and current transportation mode preferences." -o austin-congestion-study
```

This command performs two actions:
1. Creates the `austin-congestion-study/` folder structure
2. Generates a `population.v1.yaml` file containing demographic distributions

Wait for the command to complete. You should see:

```
✓ Population spec created: austin-congestion-study/population.v1.yaml
```

---

## Step 4: Navigate to the Study Folder

Change your working directory to the newly created study folder:

```bash
cd austin-congestion-study
```

Verify the folder contents:

```bash
ls -la
```

You should see:
```
population.v1.yaml
study.db
```

---

## Step 5: Create the Congestion Tax Scenario

Define the policy scenario using natural language. The Extropy engine interprets this description to generate appropriate events and behavioral triggers:

```bash
extropy scenario "The City of Austin implements a $15 daily congestion tax for vehicles entering the downtown core during peak hours (7-9 AM and 4-7 PM). The policy is announced via local news, city government communications, and social media. Commuters must decide whether to continue driving and pay the tax, shift to public transit, adjust their work schedule, or work from home." -o congestion-tax -y
```

This creates the scenario configuration at `scenario/congestion-tax/scenario.v1.yaml`.

Expected output:
```
Creating scenario: congestion-tax/scenario.v1.yaml
Base population: population.v1.yaml

✓ Found X NEW attributes
✓ Scenario created successfully
```

---

## Step 6: Generate the Persona Configuration

Create the persona rendering configuration, which determines how agents are presented during simulation:

```bash
extropy persona -s congestion-tax -y
```

This generates `scenario/congestion-tax/persona.v1.yaml`.

---

## Step 7: Sample the Agent Population

Generate 500 simulated commuters based on your population specification:

```bash
extropy sample -s congestion-tax -n 500 --seed 42
```

The `--seed 42` flag ensures reproducible results. Running this command again with the same seed will generate identical agents.

Expected output:
```
Sampling 500 agents...
✓ Sampled 500 agents to study.db
```

---

## Step 8: Generate the Social Network

Create connections between agents to simulate information spread through social ties:

```bash
extropy network -s congestion-tax --seed 42
```

This step models how awareness of the congestion tax propagates through workplace, neighborhood, and social connections.

---

## Step 9: Run the Simulation

Execute the simulation with the following command:

```bash
extropy simulate -s congestion-tax --seed 42
```

A progress indicator will display simulation status:

```
Simulating congestion-tax scenario...
[████████████████████████████████] 100%

✓ Simulation complete
  Agents processed: 500
  Timesteps: 1
  Results saved to: study.db
```

**Note:** Simulation time varies based on agent count. For 500 agents, expect 2–5 minutes.

---

## Step 10: Export the Results Report

Extract the simulation results to a JSON file:

```bash
extropy results --json > mode_shift_report.json
```

This command queries the simulation database and exports a summary report.

---

## Step 11: Verify the Deliverable

Confirm the report file exists and contains data:

```bash
ls -la mode_shift_report.json
```

To preview the contents:

```bash
head -50 mode_shift_report.json
```

The report contains aggregate statistics including:
- **Awareness rate:** Percentage of agents aware of the congestion tax
- **Attitude distribution:** Agent sentiment toward the policy
- **Behavioral outcomes:** Projected mode shift percentages (continue driving, switch to transit, work from home, etc.)

---

## Deliverable Summary

Upon completion, your `austin-congestion-study/` folder contains:

```
austin-congestion-study/
├── study.db                          # Canonical simulation database
├── population.v1.yaml                # Population specification
├── mode_shift_report.json            # ← YOUR DELIVERABLE
└── scenario/
    └── congestion-tax/
        ├── scenario.v1.yaml          # Scenario configuration
        └── persona.v1.yaml           # Persona rendering config
```

Submit `mode_shift_report.json` as evidence of completed simulation.

---

## Figures

### Figure 1: Terminal Output After Successful Simulation

*Description for screenshot:* Capture your terminal window immediately after Step 9 completes. The screenshot should show:
- The `extropy simulate` command at the top
- The progress bar at 100%
- The success message showing "Simulation complete" with agent count
- A visible command prompt ready for the next command

**Caption:** "Figure 1: Terminal output confirming successful simulation of 500 agents under the congestion tax scenario."

---

### Figure 2: Final Folder Structure with Deliverable

*Description for screenshot:* Open your file explorer (Finder on macOS, or VS Code's file tree) showing the `austin-congestion-study` folder expanded. Highlight or circle the `mode_shift_report.json` file.

**Caption:** "Figure 2: Study folder structure showing the generated mode_shift_report.json deliverable file."

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `command not found: extropy` | Run `pip install extropy-cli` again, or check your PATH |
| `ANTHROPIC_API_KEY not set` | Export your API key (see Step 2) |
| `Rate limit exceeded` | Wait 60 seconds and retry the command |
| `Simulation timeout` | Reduce agent count with `-n 250` in Step 7 |

---

## Contact

For technical support with Extropy, consult the documentation or contact the development team.
