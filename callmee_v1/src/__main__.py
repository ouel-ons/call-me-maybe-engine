"""__main__.py — CLI entry point for the call_me_maybe project.

Usage:
    uv run python -m src
    uv run python -m src --functions_definition data/input/functions_definition.json \\
                         --input data/input/function_calling_tests.json \\
                         --output data/output/function_calling_results.json
"""

import argparse
import logging
import sys

from .models import RunConfig
from .pipeline import run


def _configure_logging(verbose: bool = False) -> None:
    """Set up root logger with a human-friendly format.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: list) -> RunConfig:
    """Parse command-line arguments into a RunConfig.

    Args:
        argv: sys.argv[1:] or a test list.

    Returns:
        Populated RunConfig instance.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="LLM function calling with constrained decoding.",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default="data/input/function_calling_tests.json",
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="data/output/function_calling_results.json",
        help="Path to write the results JSON file.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=None,
        help="Override the default model (must still work with Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    return RunConfig(
        functions_definition=args.functions_definition,
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
    )


def main(argv: list = sys.argv[1:]) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # Parse args first so we know the verbose flag.
    parsed = None
    try:
        config = _parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1

    _configure_logging(verbose="--verbose" in argv or "-v" in argv)
    logger = logging.getLogger(__name__)

    logger.info("Starting call-me-maybe pipeline")
    logger.info("  Functions: %s", config.functions_definition)
    logger.info("  Input:     %s", config.input_path)
    logger.info("  Output:    %s", config.output_path)

    try:
        results = run(config)
        logger.info("Pipeline complete — %d results written.", len(results))
        return 0
    except Exception as exc:
        logger.error("Pipeline failed with unexpected error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
