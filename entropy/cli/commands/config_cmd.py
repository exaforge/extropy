"""Config command for viewing and managing entropy configuration."""

import typer

from ..app import app, console
from ...config import (
    get_config,
    reset_config,
    CONFIG_FILE,
    get_api_key,
    get_azure_config,
)


VALID_KEYS = {
    "pipeline.provider",
    "pipeline.model_simple",
    "pipeline.model_reasoning",
    "pipeline.model_research",
    "simulation.provider",
    "simulation.model",
    "simulation.pivotal_model",
    "simulation.routine_model",
    "simulation.max_concurrent",
    "simulation.rate_tier",
    "simulation.rpm_override",
    "simulation.tpm_override",
}

INT_FIELDS = {"max_concurrent", "rate_tier", "rpm_override", "tpm_override"}


@app.command("config")
def config_command(
    action: str = typer.Argument(
        ...,
        help="Action: show, set, reset",
    ),
    key: str | None = typer.Argument(
        None,
        help="Config key (e.g. pipeline.provider, simulation.model)",
    ),
    value: str | None = typer.Argument(
        None,
        help="Value to set",
    ),
):
    """View or modify entropy configuration.

    Examples:
        entropy config show
        entropy config set pipeline.provider claude
        entropy config set simulation.provider openai
        entropy config set simulation.model gpt-5-mini
        entropy config reset
    """
    if action == "show":
        _show_config()
    elif action == "set":
        if not key or value is None:
            console.print("[red]Usage:[/red] entropy config set <key> <value>")
            console.print()
            console.print("Available keys:")
            for k in sorted(VALID_KEYS):
                console.print(f"  {k}")
            raise typer.Exit(1)
        _set_config(key, value)
    elif action == "reset":
        _reset_config()
    else:
        console.print(f"[red]Unknown action:[/red] {action}")
        console.print("Valid actions: show, set, reset")
        raise typer.Exit(1)


def _show_config():
    """Display current resolved configuration."""
    config = get_config()

    console.print()
    console.print("[bold]Entropy Configuration[/bold]")
    console.print("─" * 40)

    # Pipeline zone
    console.print()
    console.print("[bold cyan]Pipeline[/bold cyan] (spec, extend, persona, scenario)")
    console.print(f"  provider        = {config.pipeline.provider}")
    console.print(
        f"  model_simple    = {config.pipeline.model_simple or '[dim](provider default)[/dim]'}"
    )
    console.print(
        f"  model_reasoning = {config.pipeline.model_reasoning or '[dim](provider default)[/dim]'}"
    )
    console.print(
        f"  model_research  = {config.pipeline.model_research or '[dim](provider default)[/dim]'}"
    )

    # Simulation zone
    console.print()
    console.print("[bold cyan]Simulation[/bold cyan] (agent reasoning)")
    console.print(f"  provider        = {config.simulation.provider}")
    console.print(
        f"  model           = {config.simulation.model or '[dim](provider default)[/dim]'}"
    )
    console.print(
        f"  pivotal_model   = {config.simulation.pivotal_model or '[dim](same as model)[/dim]'}"
    )
    console.print(
        f"  routine_model   = {config.simulation.routine_model or '[dim](provider default)[/dim]'}"
    )
    console.print(f"  max_concurrent  = {config.simulation.max_concurrent}")
    console.print(
        f"  rate_tier       = {config.simulation.rate_tier or '[dim](tier 1)[/dim]'}"
    )
    if config.simulation.rpm_override:
        console.print(f"  rpm_override    = {config.simulation.rpm_override}")
    if config.simulation.tpm_override:
        console.print(f"  tpm_override    = {config.simulation.tpm_override}")

    # API keys status
    console.print()
    console.print("[bold cyan]API Keys[/bold cyan] (from env vars)")
    _show_key_status("openai", "OPENAI_API_KEY")
    _show_key_status("claude", "ANTHROPIC_API_KEY")
    _show_key_status("azure_openai", "AZURE_OPENAI_API_KEY")

    # Azure-specific config (show when Azure provider is in use)
    active_providers = {config.pipeline.provider, config.simulation.provider}
    if "azure_openai" in active_providers:
        azure_cfg = get_azure_config("azure_openai")
        console.print()
        console.print("[bold cyan]Azure OpenAI[/bold cyan]")
        console.print(
            f"  endpoint          = {azure_cfg['azure_endpoint'] or '[dim]not set[/dim]'}"
        )
        console.print(f"  api_version       = {azure_cfg['api_version']}")
        if azure_cfg["azure_deployment"]:
            console.print(f"  deployment        = {azure_cfg['azure_deployment']}")

    # Config file
    console.print()
    if CONFIG_FILE.exists():
        console.print(f"Config file: {CONFIG_FILE}")
    else:
        console.print(f"Config file: [dim]not created yet[/dim] ({CONFIG_FILE})")
    console.print()


def _show_key_status(provider: str, env_var_label: str):
    """Show whether an API key is configured."""
    key = get_api_key(provider)
    if key:
        masked = key[:8] + "..." + key[-4:] if len(key) > 16 else "***"
        console.print(f"  {env_var_label}: [green]{masked}[/green]")
    else:
        console.print(f"  {env_var_label}: [dim]not set[/dim]")


def _set_config(key: str, value: str):
    """Set a config value and save."""
    if key not in VALID_KEYS:
        console.print(f"[red]Unknown key:[/red] {key}")
        console.print()
        console.print("Available keys:")
        for k in sorted(VALID_KEYS):
            console.print(f"  {k}")
        raise typer.Exit(1)

    # Load current config (or defaults if no file)
    config = get_config()

    zone, field = key.split(".", 1)
    if zone == "pipeline":
        target = config.pipeline
    elif zone == "simulation":
        target = config.simulation
    else:
        console.print(f"[red]Unknown zone:[/red] {zone}")
        raise typer.Exit(1)

    # Type coercion
    if field in INT_FIELDS:
        try:
            setattr(target, field, int(value))
        except ValueError:
            console.print(f"[red]Invalid integer value:[/red] {value}")
            raise typer.Exit(1)
    else:
        setattr(target, field, value)

    config.save()
    reset_config()  # Clear cached singleton so next get_config() reloads

    console.print(f"[green]✓[/green] Set {key} = {value}")
    console.print(f"  Saved to {CONFIG_FILE}")


def _reset_config():
    """Reset config to defaults."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        reset_config()
        console.print("[green]✓[/green] Config reset to defaults")
        console.print(f"  Removed {CONFIG_FILE}")
    else:
        console.print("Config already at defaults (no config file exists)")
