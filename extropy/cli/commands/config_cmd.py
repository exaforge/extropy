"""Config command for viewing and managing extropy configuration."""

import typer

from ..app import app, console
from ...config import (
    get_config,
    reset_config,
    CONFIG_FILE,
    get_api_key_for_provider,
)


VALID_KEYS = {
    "models.fast",
    "models.strong",
    "simulation.fast",
    "simulation.strong",
    "simulation.max_concurrent",
    "simulation.rate_tier",
    "simulation.rpm_override",
    "simulation.tpm_override",
    "defaults.population_size",
    "defaults.db_path",
}

INT_FIELDS = {
    "max_concurrent",
    "rate_tier",
    "rpm_override",
    "tpm_override",
    "population_size",
}


@app.command("config")
def config_command(
    action: str = typer.Argument(
        ...,
        help="Action: show, set, reset",
    ),
    key: str | None = typer.Argument(
        None,
        help="Config key (e.g. models.fast, simulation.strong)",
    ),
    value: str | None = typer.Argument(
        None,
        help="Value to set",
    ),
):
    """View or modify extropy configuration.

    Examples:
        extropy config show
        extropy config set models.fast openai/gpt-5-mini
        extropy config set models.strong anthropic/claude-sonnet-4.5
        extropy config set simulation.strong openrouter/anthropic/claude-sonnet-4.5
        extropy config reset
    """
    if action == "show":
        _show_config()
    elif action == "set":
        if not key or value is None:
            console.print("[red]Usage:[/red] extropy config set <key> <value>")
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
    console.print("[bold]Extropy Configuration[/bold]")
    console.print("─" * 40)

    # Models (pipeline)
    console.print()
    console.print(
        "[bold cyan]Models[/bold cyan] (pipeline: spec, extend, persona, scenario)"
    )
    console.print(f"  fast   = {config.models.fast}")
    console.print(f"  strong = {config.models.strong}")

    # Simulation
    console.print()
    console.print("[bold cyan]Simulation[/bold cyan] (agent reasoning)")
    strong_val = config.simulation.strong or "[dim](= models.strong)[/dim]"
    fast_val = config.simulation.fast or "[dim](= models.fast)[/dim]"
    console.print(f"  strong          = {strong_val}")
    console.print(f"  fast            = {fast_val}")
    console.print(f"  max_concurrent  = {config.simulation.max_concurrent}")
    console.print(
        f"  rate_tier       = {config.simulation.rate_tier or '[dim](tier 1)[/dim]'}"
    )
    if config.simulation.rpm_override:
        console.print(f"  rpm_override    = {config.simulation.rpm_override}")
    if config.simulation.tpm_override:
        console.print(f"  tpm_override    = {config.simulation.tpm_override}")

    # Custom providers
    if config.providers:
        console.print()
        console.print("[bold cyan]Custom Providers[/bold cyan]")
        for name, provider_cfg in config.providers.items():
            console.print(f"  {name}:")
            console.print(f"    base_url    = {provider_cfg.base_url}")
            if provider_cfg.api_key_env:
                console.print(f"    api_key_env = {provider_cfg.api_key_env}")

    # Defaults
    console.print()
    console.print("[bold cyan]Defaults[/bold cyan]")
    console.print(f"  population_size = {config.defaults.population_size}")
    console.print(f"  db_path         = {config.defaults.db_path}")

    # API keys status
    console.print()
    console.print("[bold cyan]API Keys[/bold cyan] (from env vars)")
    _show_key_status("openai", "OPENAI_API_KEY")
    _show_key_status("anthropic", "ANTHROPIC_API_KEY")
    _show_key_status("azure", "AZURE_OPENAI_API_KEY")
    _show_key_status("openrouter", "OPENROUTER_API_KEY")
    _show_key_status("deepseek", "DEEPSEEK_API_KEY")

    # Config file
    console.print()
    if CONFIG_FILE.exists():
        console.print(f"Config file: {CONFIG_FILE}")
    else:
        console.print(f"Config file: [dim]not created yet[/dim] ({CONFIG_FILE})")
    console.print()


def _show_key_status(provider: str, env_var_label: str):
    """Show whether an API key is configured."""
    key = get_api_key_for_provider(provider)
    if key:
        masked = key[:8] + "..." + key[-4:] if len(key) > 16 else "***"
        console.print(f"  {env_var_label}: [green]{masked}[/green]")
    else:
        console.print(f"  {env_var_label}: [dim]not set[/dim]")


def _set_config(key: str, value: str):
    """Set a config value and save."""
    # Allow dynamic provider keys like providers.mycompany.base_url
    is_provider_key = key.startswith("providers.")
    if key not in VALID_KEYS and not is_provider_key:
        console.print(f"[red]Unknown key:[/red] {key}")
        console.print()
        console.print("Available keys:")
        for k in sorted(VALID_KEYS):
            console.print(f"  {k}")
        console.print("  providers.<name>.base_url")
        console.print("  providers.<name>.api_key_env")
        raise typer.Exit(1)

    # Load current config (or defaults if no file)
    config = get_config()

    if is_provider_key:
        parts = key.split(".", 2)
        if len(parts) != 3 or parts[2] not in ("base_url", "api_key_env"):
            console.print(
                f"[red]Invalid provider key:[/red] {key}\n"
                "Expected: providers.<name>.base_url or providers.<name>.api_key_env"
            )
            raise typer.Exit(1)
        provider_name = parts[1]
        field = parts[2]
        from ...config import CustomProviderConfig

        if provider_name not in config.providers:
            config.providers[provider_name] = CustomProviderConfig()
        setattr(config.providers[provider_name], field, value)
    else:
        zone, field_name = key.split(".", 1)
        if zone == "models":
            target = config.models
        elif zone == "simulation":
            target = config.simulation
        elif zone == "defaults":
            target = config.defaults
        else:
            console.print(f"[red]Unknown zone:[/red] {zone}")
            raise typer.Exit(1)

        # Type coercion
        if field_name in INT_FIELDS:
            try:
                setattr(target, field_name, int(value))
            except ValueError:
                console.print(f"[red]Invalid integer value:[/red] {value}")
                raise typer.Exit(1)
        else:
            setattr(target, field_name, value)

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
