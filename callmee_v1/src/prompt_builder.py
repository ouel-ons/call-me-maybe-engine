"""prompt_builder.py — Build the prompt that the LLM uses to select a function.

The prompt does two jobs:
1. Tells the model about the available functions (so it understands the domain).
2. Asks it to produce a JSON function call for the given user request.

We do NOT rely on the model to spontaneously produce valid JSON — that is
handled by the constrained decoder. The prompt just provides semantic context.
"""

from typing import List
from .models import FunctionDefinition


_SYSTEM_HEADER = (
    "You are a function-calling assistant. "
    "Given a user request and a list of available functions, "
    "output ONLY a JSON object with keys \"function\" and \"arguments\".\n\n"
)


def build_prompt(
    user_request: str,
    functions: List[FunctionDefinition],
) -> str:
    """Construct the full prompt string for the LLM.

    Args:
        user_request: The natural-language prompt from the test file.
        functions: The list of available function definitions.

    Returns:
        A single string ready for tokenisation.
    """
    lines: List[str] = [_SYSTEM_HEADER, "Available functions:\n"]

    for fn in functions:
        lines.append(f"  - {fn.name}: {fn.description}")
        for param_name, param_info in fn.parameters.items():
            lines.append(f"      {param_name} ({param_info.type})")

    lines.append("\nUser request: " + user_request)
    lines.append('\nJSON output: {"function": "')  # prime the model

    return "".join(lines)
