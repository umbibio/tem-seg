"""CLI commands for working with configuration presets."""

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tem_analysis_pipeline.configuration.config import OrganelleType, WorkflowConfig
from tem_analysis_pipeline.configuration.presets import get_preset_for_organelle
from tem_analysis_pipeline.configuration.validation import (
    ConfigValidator,
    ValidationError,
    ValidationLevel,
)


# OrganelleType is now directly usable with CLI since it inherits from str
# and has CLI-friendly values


def create_config_app() -> typer.Typer:
    """Create the configuration CLI application.

    Returns:
        Typer app for configuration commands.
    """
    app = typer.Typer(help="Work with configuration presets")

    @app.command("list")
    def list_presets():
        """List available configuration presets."""
        console = Console()
        table = Table(title="Available Configuration Presets")

        table.add_column("Organelle", style="green")
        table.add_column("CLI Option", style="blue")
        table.add_column("Description", style="yellow")

        table.add_row(
            "Mitochondria", "mitochondria", "Optimized for mitochondria segmentation"
        )
        table.add_row("Nucleus", "nucleus", "Optimized for nucleus segmentation")
        table.add_row(
            "Endoplasmic Reticulum",
            "er",
            "Optimized for endoplasmic reticulum segmentation",
        )

        console.print(table)

    @app.command("generate")
    def generate_preset(
        organelle: OrganelleType = typer.Argument(
            ...,
            help="Organelle type to generate a preset for",
        ),
        name: Optional[str] = typer.Option(
            None,
            "--name",
            "-n",
            help="Name for the configuration (defaults to organelle name)",
        ),
        output: Path = typer.Option(
            "config.yaml",
            "--output",
            "-o",
            help="Output file path for the configuration",
        ),
        validate: bool = typer.Option(
            True,
            "--validate/--no-validate",
            help="Validate the configuration before saving",
        ),
    ):
        """Generate a configuration preset for a specific organelle.

        This creates an optimized configuration for the specified organelle
        type and saves it to a YAML file.
        """
        console = Console()

        try:
            # Use OrganelleType directly - no mapping needed

            # Generate the preset
            preset = get_preset_for_organelle(organelle, name)

            # Validate if requested
            if validate:
                validator = ConfigValidator(level=ValidationLevel.STRICT)
                validator.validate(preset)
                console.print("[green]Configuration validated successfully[/green]")

            # Save the configuration
            preset.save(output)

            # Display success message with details
            rprint(
                Panel.fit(
                    f"[bold green]Configuration preset generated![/bold green]\n\n"
                    f"Organelle: [yellow]{organelle.name}[/yellow]\n"
                    f"Name: [blue]{preset.name}[/blue]\n"
                    f"Output file: [cyan]{output.absolute()}[/cyan]"
                )
            )

        except ValidationError as e:
            console.print(f"[bold red]Validation error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    @app.command("validate")
    def validate_config(
        config_file: Path = typer.Argument(
            ...,
            help="Path to a YAML configuration file to validate",
            exists=True,
        ),
        strict: bool = typer.Option(
            True,
            "--strict/--lenient",
            help="Use strict validation (fail on any error) or lenient (show warnings only)",
        ),
    ):
        """Validate a configuration file.

        Checks if a YAML configuration file meets all requirements
        and constraints for TEM-Seg workflows.
        """
        console = Console()

        try:
            # Load the configuration
            config = WorkflowConfig.from_file(config_file)

            # Create validator with appropriate level
            level = ValidationLevel.STRICT if strict else ValidationLevel.WARNING
            validator = ConfigValidator(level=level)

            # Validate the configuration
            if strict:
                try:
                    validator.validate(config)
                    console.print("[bold green]Configuration is valid![/bold green]")
                except ValidationError as e:
                    console.print(f"[bold red]Validation error:[/bold red] {str(e)}")
                    raise typer.Exit(code=1)
            else:
                errors = validator.validate(config)
                if errors:
                    console.print("[bold yellow]Validation warnings:[/bold yellow]")
                    for error in errors:
                        console.print(f"  - {str(error)}")
                else:
                    console.print("[bold green]Configuration is valid![/bold green]")

            # Display configuration summary
            table = Table(title=f"Configuration Summary: {config.name}")
            table.add_column("Property", style="blue")
            table.add_column("Value", style="green")

            table.add_row("Name", config.name)
            table.add_row("Organelle Type", config.organelle_type.name)
            table.add_row("Version", config.version)
            table.add_row("Model Architecture", config.model.architecture)
            table.add_row("Training Epochs", str(config.training.epochs))
            table.add_row("Configuration Hash", config.get_hash()[:8])

            console.print(table)

        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] File not found: {config_file}")
            raise typer.Exit(code=1)

        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    return app
