"""io_helpers.py — JSON file loading and saving with robust error handling.

Every I/O operation in this module catches exceptions and provides clear error
messages rather than letting the program crash unexpectedly.
"""

import json
import logging
import os
from typing import Any, List

from pydantic import ValidationError

from .models import FunctionDefinition, FunctionCall, Prompt

logger = logging.getLogger(__name__)


def load_function_definitions(path: str) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions_definition.json file.

    Returns:
        List of validated FunctionDefinition objects.
        Returns empty list and logs an error on failure.
    """
    raw = _load_json_file(path)
    if raw is None:
        return []

    if not isinstance(raw, list):
        logger.error(
            "functions_definition file must contain a JSON array, got %s",
            type(raw).__name__,
        )
        return []

    definitions: List[FunctionDefinition] = []
    for i, item in enumerate(raw):
        try:
            definitions.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            logger.warning(
                "Skipping function definition #%d due to validation error: %s",
                i,
                exc,
            )

    logger.info("Loaded %d function definitions from %s", len(definitions), path)
    return definitions


def load_prompts(path: str) -> List[Prompt]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the function_calling_tests.json file.

    Returns:
        List of validated Prompt objects.
        Returns empty list and logs an error on failure.
    """
    raw = _load_json_file(path)
    if raw is None:
        return []

    if not isinstance(raw, list):
        logger.error(
            "Input file must contain a JSON array, got %s",
            type(raw).__name__,
        )
        return []

    prompts: List[Prompt] = []
    for i, item in enumerate(raw):
        try:
            prompts.append(Prompt.model_validate(item))
        except ValidationError as exc:
            logger.warning(
                "Skipping prompt #%d due to validation error: %s", i, exc
            )

    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


def save_results(results: List[FunctionCall], path: str) -> bool:
    """Serialise and write results to a JSON file.

    Args:
        results: List of FunctionCall objects to serialise.
        path: Destination file path.

    Returns:
        True on success, False on failure.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    serialised = [r.model_dump() for r in results]

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serialised, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote %d results to %s", len(results), path)
        return True
    except OSError as exc:
        logger.error("Failed to write results to %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json_file(path: str) -> Any:
    """Load a JSON file and return the parsed object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object, or None on any error.
    """
    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.error("Could not read %s: %s", path, exc)
        return None
