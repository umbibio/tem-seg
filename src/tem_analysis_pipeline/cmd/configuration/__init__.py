"""Configuration system CLI commands."""

import typer

from tem_analysis_pipeline.cmd.configuration.config_commands import create_config_app

app = typer.Typer(help="Configuration management commands")

# Register subcommands
app.add_typer(
    create_config_app(), name="preset", help="Work with configuration presets"
)

__all__ = ["app"]
