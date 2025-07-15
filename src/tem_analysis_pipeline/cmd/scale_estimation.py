"""Command line interface for Scale Estimation System."""

from pathlib import Path
from typing import Annotated, List, Optional

import typer
from typer import Argument, Option

from ..scale_estimation import ScaleBarReader, ScaleManager


def create_scale_app() -> typer.Typer:
    """Create the scale estimation command line interface."""
    scale_app = typer.Typer(help="Scale Estimation for TEM images")

    @scale_app.command("get-scale")
    def get_scale_command(
        image_paths: Annotated[
            List[Path], Argument(help="Path(s) to TEM image(s) for scale detection")
        ],
        fallback: Annotated[
            Optional[float],
            Option(
                "--fallback", "-f", help="Fallback scale in μm/pixel if detection fails"
            ),
        ] = None,
        output: Annotated[
            Optional[Path],
            Option(
                "--output",
                "-o",
                help="Path to output CSV file with image paths and detected scales",
            ),
        ] = None,
        verbose: Annotated[
            bool,
            Option(
                "--verbose", "-v", help="Print detailed information during processing"
            ),
        ] = False,
        debug: Annotated[
            bool,
            Option(
                "--debug",
                "-d",
                help="Print extensive debug information during processing",
            ),
        ] = False,
        skip_missing: Annotated[
            bool,
            Option(
                "--skip-missing",
                "-s",
                help="Skip images without scale bars instead of reporting errors",
            ),
        ] = False,
    ) -> None:
        """Detect scale in TEM images using the Scale Estimation System.

        This command will detect the scale bar in TEM images and return the
        scale in micrometers per pixel. If multiple images are provided,
        it will process all of them and optionally save the results to a CSV file.
        """
        import csv
        import sys
        import traceback

        from ..scale_estimation import (
            InvalidInputError,
            NoScaleFoundError,
            ScaleDetectionFailureError,
            ScaleEstimationError,
        )

        # Override print function in calibration module if debug is enabled
        if debug:
            from .. import calibration

            original_print = print

            def debug_print(*args, **kwargs):
                typer.echo(f"[DEBUG] {' '.join(str(arg) for arg in args)}")
                original_print(*args, **kwargs)

            calibration.print = debug_print

        # Set up ScaleManager with ScaleBarReader
        reader = ScaleBarReader()
        manager = ScaleManager([reader])
        results = []

        # Process each image
        for img_path in image_paths:
            if not img_path.exists():
                typer.echo(f"Error: Image file not found: {img_path}")
                continue

            try:
                if verbose or debug:
                    typer.echo(f"Processing {img_path}...")

                if debug:
                    typer.echo("[DEBUG] Opening image and converting to PIL format")

                # Try to get scale, use fallback if provided
                if debug:
                    # For debugging, use the ScaleBarReader directly to get more information
                    try:
                        img = reader._convert_to_pil(img_path)
                        typer.echo(
                            f"[DEBUG] Image loaded: size={img.size}, mode={img.mode}"
                        )
                        typer.echo("[DEBUG] Converting to grayscale if needed")
                        if img.mode != "L":
                            img = img.convert("L")
                            typer.echo(f"[DEBUG] Converted to mode={img.mode}")
                        typer.echo("[DEBUG] Attempting to detect scale bar")
                        scale = reader.estimate_scale(img)
                        typer.echo("[DEBUG] Scale bar detected successfully")
                        status = "success"  # Mark as successful detection
                    except Exception as e:
                        typer.echo(f"[DEBUG] Error in direct ScaleBarReader: {str(e)}")
                        typer.echo(
                            f"[DEBUG] Traceback: {''.join(traceback.format_tb(sys.exc_info()[2]))}"
                        )
                        raise
                else:
                    # Try to get scale from the manager, which handles fallback internally
                    scale = manager.get_scale(img_path, fallback=fallback)

                    # Check if fallback was used by comparing the result with the fallback value
                    if fallback is not None and abs(scale - fallback) < 1e-9:
                        status = "fallback_used"  # Mark this as using fallback
                    else:
                        status = "success"

                # Store result
                results.append(
                    {"image": str(img_path), "scale": scale, "status": status}
                )

                if verbose or debug:
                    typer.echo(f"Scale detected: {scale} μm/pixel")

            except NoScaleFoundError as e:
                # More specific error for when no scale bar is found
                if fallback is not None:
                    # This should never happen here since the manager would have used the fallback
                    # But we keep it as a safety measure
                    results.append(
                        {
                            "image": str(img_path),
                            "scale": fallback,
                            "status": "fallback_used",
                        }
                    )
                    typer.echo(
                        f"No scale bar found in {img_path}, using fallback: {fallback} μm/pixel"
                    )
                elif skip_missing:
                    results.append(
                        {
                            "image": str(img_path),
                            "scale": None,
                            "status": "skipped_no_scale",
                        }
                    )
                    typer.echo(
                        f"No scale bar found in {img_path}, skipping (--skip-missing enabled)"
                    )
                else:
                    # Report as error when no fallback and not skipping
                    status = "no_scale_found"
                    results.append(
                        {"image": str(img_path), "scale": None, "status": status}
                    )
                    typer.echo(f"No scale bar found in {img_path}: {e}")

                if debug:
                    typer.echo(f"[DEBUG] Detailed error: {str(e)}")
                    typer.echo(
                        "[DEBUG] This typically means the image doesn't contain a recognized scale bar format"
                    )

            except ScaleDetectionFailureError as e:
                # Error when scale detection process fails
                status = "detection_failure"
                results.append(
                    {"image": str(img_path), "scale": None, "status": status}
                )
                typer.echo(f"Scale detection failed for {img_path}: {e}")
                if debug:
                    typer.echo(f"[DEBUG] Detailed error: {str(e)}")
                    typer.echo(
                        f"[DEBUG] Traceback: {''.join(traceback.format_tb(sys.exc_info()[2]))}"
                    )

            except InvalidInputError as e:
                # Error for invalid input
                status = "invalid_input"
                results.append(
                    {"image": str(img_path), "scale": None, "status": status}
                )
                typer.echo(f"Invalid input for {img_path}: {e}")

            except ScaleEstimationError as e:
                # Handle other scale estimation errors
                if fallback is not None:
                    results.append(
                        {
                            "image": str(img_path),
                            "scale": fallback,
                            "status": "fallback_used",
                        }
                    )
                    typer.echo(
                        f"Failed to detect scale in {img_path}, using fallback: {fallback} μm/pixel"
                    )
                elif skip_missing:
                    # Also apply skip_missing to general scale estimation errors
                    results.append(
                        {
                            "image": str(img_path),
                            "scale": None,
                            "status": "skipped_no_scale",
                        }
                    )
                    typer.echo(
                        f"Failed to detect scale in {img_path}, skipping (--skip-missing enabled)"
                    )
                else:
                    # Report as error when no fallback and not skipping
                    status = "error"
                    results.append(
                        {"image": str(img_path), "scale": None, "status": status}
                    )
                    typer.echo(f"Error processing {img_path}: {e}")

            except Exception as e:
                # Unexpected error
                status = "unexpected_error"
                results.append(
                    {
                        "image": str(img_path),
                        "scale": None,
                        "status": f"unexpected_error: {e}",
                    }
                )
                typer.echo(f"Unexpected error processing {img_path}: {e}")
                if debug:
                    typer.echo(
                        f"[DEBUG] Traceback: {''.join(traceback.format_tb(sys.exc_info()[2]))}"
                    )

        # Compile statistics
        success_count = sum(1 for r in results if r["status"] == "success")
        fallback_count = sum(1 for r in results if r["status"] == "fallback_used")
        skipped_count = sum(1 for r in results if r["status"] == "skipped_no_scale")
        error_count = len(results) - success_count - fallback_count - skipped_count

        # Print summary
        typer.echo("\nSummary:")
        typer.echo(
            f"  Successfully detected: {success_count}/{len(image_paths)} images"
        )
        if fallback_count > 0:
            typer.echo(
                f"  Using fallback scale: {fallback_count}/{len(image_paths)} images"
            )
        if skipped_count > 0:
            typer.echo(
                f"  Skipped (no scale): {skipped_count}/{len(image_paths)} images"
            )
        if error_count > 0:
            typer.echo(f"  Failed to process: {error_count}/{len(image_paths)} images")

        if fallback_count > 0 and fallback is not None:
            typer.echo(f"\nFallback scale used: {fallback} μm/pixel")

        if success_count == 0 and fallback_count == 0 and not skip_missing:
            typer.echo(
                "\nTip: Use --fallback VALUE to provide a default scale when detection fails"
            )
            typer.echo("     Use --skip-missing to ignore images without scale bars")
            typer.echo("     Use --debug for more detailed error information")

        # Save results if output path is provided
        if output and results:
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "scale", "status"])
                writer.writeheader()
                writer.writerows(results)
                typer.echo(f"Results saved to {output}")

        # Print results to console if only one image was processed
        if len(results) == 1:
            if results[0]["status"] == "success":
                typer.echo(f"Scale: {results[0]['scale']} μm/pixel")
            elif results[0]["status"] == "fallback_used":
                typer.echo(f"Scale (fallback): {results[0]['scale']} μm/pixel")

    return scale_app
