# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration, loaded from environment variables."""

from pathlib import Path
from typing import Any

from graphrag.config.models.graph_rag_config import GraphRagConfig


def create_graphrag_config(
    values: dict[str, Any] | None = None,
    root_dir: str | None = None,
) -> GraphRagConfig:
    """Load Configuration Parameters from a dictionary.

    Parameters
    ----------
    values : dict[str, Any] | None
        Dictionary of configuration values to pass into pydantic model.
    root_dir : str | None
        Root directory for the project.
    skip_validation : bool
        Skip pydantic model validation of the configuration.
        This is useful for testing and mocking purposes but
        should not be used in the core code or API.

    Returns
    -------
    GraphRagConfig
        The configuration object.

    Raises
    ------
    ValidationError
        If the configuration values do not satisfy pydantic validation.
    """
    values = values or {}
    if root_dir:
        root_path = Path(root_dir).resolve()
        values["root_dir"] = str(root_path)

    # If the configuration intends to use API key authentication, attempt to
    # populate missing API key fields from common environment variables so
    # that GraphRagConfig validation does not fail with unclear errors.
    # We attempt several common names to be robust across environments.
    auth = values.get("auth") or values.get("authentication") or values.get("auth_type")
    if auth == "api_key":
        import os

        # Populate a generic api_key if not present
        values.setdefault(
            "api_key",
            os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )

        # Populate chat-specific API key if not present
        values.setdefault(
            "chat_api_key",
            os.environ.get("GRAPHRAG_CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_CHAT_API_KEY")
        )

        # Populate embedding-specific API key if not present
        values.setdefault(
            "embedding_api_key",
            os.environ.get("GRAPHRAG_EMBEDDING_API_KEY") or os.environ.get("OPENAI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )

    return GraphRagConfig(**values)
