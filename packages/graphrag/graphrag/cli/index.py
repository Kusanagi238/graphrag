# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI implementation of the index subcommand."""

import asyncio
import logging
import sys
import warnings
from pathlib import Path

import graphrag.api as api
from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.config.enums import CacheType, IndexingMethod
from graphrag.config.load_config import load_config
from graphrag.index.validate_config import validate_config_names
from graphrag.utils.cli import redact

# Ignore warnings from numba
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

logger = logging.getLogger(__name__)


def _register_signal_handlers():
    import signal

    def handle_signal(signum, _):
        # Handle the signal here
        logger.debug(f"Received signal {signum}, exiting...")  # noqa: G004
        for task in asyncio.all_tasks():
            task.cancel()
        logger.debug("All tasks cancelled. Exiting...")

    # Register signal handlers for SIGINT and SIGHUP
    signal.signal(signal.SIGINT, handle_signal)

    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, handle_signal)


def index_cli(
    root_dir: Path,
    method: IndexingMethod,
    verbose: bool,
    memprofile: bool,
    cache: bool,
    config_filepath: Path | None,
    dry_run: bool,
    skip_validation: bool,
    output_dir: Path | None,
):
    """Run the pipeline with the given config."""
    cli_overrides = {}
    if output_dir:
        cli_overrides["output.base_dir"] = str(output_dir)
        cli_overrides["reporting.base_dir"] = str(output_dir)
        cli_overrides["update_index_output.base_dir"] = str(output_dir)
    try:
        config = load_config(root_dir, config_filepath, cli_overrides)
    except Exception as e:
        handled = False
        try:
            from pydantic import ValidationError as PydanticValidationError
            if isinstance(e, PydanticValidationError):
                handled = True
        except Exception:
            # Fall back to name/string matching if pydantic isn't importable
            if e.__class__.__name__.endswith("ValidationError") or "API Key is required" in str(e):
                handled = True
        if handled:
            logger.error("Configuration validation failed: %s", e)
            print(
                "Configuration error: API Key is required for chat/embedding when using api_key authentication. "
                "Please set the API key in your config file, provide it via environment variables, or pass it via CLI overrides."
            )
            import sys

            sys.exit(2)
        raise
    _run_index(
        config=config,
        method=method,
        is_update_run=False,
        verbose=verbose,
        memprofile=memprofile,
        cache=cache,
        dry_run=dry_run,
        skip_validation=skip_validation,
    )


def update_cli(
    root_dir: Path,
    method: IndexingMethod,
    verbose: bool,
    memprofile: bool,
    cache: bool,
    config_filepath: Path | None,
    skip_validation: bool,
    output_dir: Path | None,
):
    """Run the pipeline with the given config."""
    cli_overrides = {}
    if output_dir:
        cli_overrides["output.base_dir"] = str(output_dir)
        cli_overrides["reporting.base_dir"] = str(output_dir)
        cli_overrides["update_index_output.base_dir"] = str(output_dir)

    try:
        config = load_config(root_dir, config_filepath, cli_overrides)
    except Exception as e:
        handled = False
        try:
            from pydantic import ValidationError as PydanticValidationError
            if isinstance(e, PydanticValidationError):
                handled = True
        except Exception:
            # Fall back to name/string matching if pydantic isn't importable
            if e.__class__.__name__.endswith("ValidationError") or "API Key is required" in str(e):
                handled = True
        if handled:
            logger.error("Configuration validation failed: %s", e)
            print(
                "Configuration error: API Key is required for chat/embedding when using api_key authentication. "
                "Please set the API key in your config file, provide it via environment variables, or pass it via CLI overrides."
            )
            import sys

            sys.exit(2)
        raise

    _run_index(
        config=config,
        method=method,
        is_update_run=True,
        verbose=verbose,
        memprofile=memprofile,
        cache=cache,
        dry_run=False,
        skip_validation=skip_validation,
    )


def _run_index(
    config,
    method,
    is_update_run,
    verbose,
    memprofile,
    cache,
    dry_run,
    skip_validation,
):
    # Configure the root logger with the specified log level
    from graphrag.logger.standard_logging import init_loggers

    # Initialize loggers and reporting config
    init_loggers(
        config=config,
        verbose=verbose,
    )

    if not cache:
        config.cache.type = CacheType.none

    if not skip_validation:
        validate_config_names(config)

    logger.info("Starting pipeline run. %s", dry_run)
    logger.info(
        "Using default configuration: %s",
        redact(config.model_dump()),
    )

    if dry_run:
        logger.info("Dry run complete, exiting...", True)
        sys.exit(0)

    _register_signal_handlers()

    outputs = asyncio.run(
        api.build_index(
            config=config,
            method=method,
            is_update_run=is_update_run,
            memory_profile=memprofile,
            callbacks=[ConsoleWorkflowCallbacks(verbose=verbose)],
            verbose=verbose,
        )
    )
    encountered_errors = any(
        output.errors and len(output.errors) > 0 for output in outputs
    )

    if encountered_errors:
        logger.error(
            "Errors occurred during the pipeline run, see logs for more details."
        )
    else:
        logger.info("All workflows completed successfully.")

    sys.exit(1 if encountered_errors else 0)
