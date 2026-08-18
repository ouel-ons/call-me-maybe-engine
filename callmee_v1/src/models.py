"""models.py — Pydantic v2 models for function definitions and results.

All classes use pydantic for validation as required by the project spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Input models: function definitions
# ---------------------------------------------------------------------------

class ParameterType(BaseModel):
    """Type descriptor for a single function parameter.

    Attributes:
        type: JSON-schema-style type string (number, string, boolean, etc.).
    """

    model_config = ConfigDict(extra="allow")

    type: str


class ReturnType(BaseModel):
    """Return type descriptor for a function definition.

    Attributes:
        type: JSON-schema-style type string.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class FunctionDefinition(BaseModel):
    """A single callable function exposed to the LLM.

    Attributes:
        name: Unique function identifier (e.g. fn_add_numbers).
        description: Natural-language description used in the prompt.
        parameters: Mapping of parameter name → ParameterType.
        returns: Return type descriptor.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ReturnType

    @field_validator("name")
    @classmethod
    def name_must_be_non_empty(cls, v: str) -> str:
        """Validate that the function name is not blank."""
        if not v.strip():
            raise ValueError("Function name must not be empty")
        return v


# ---------------------------------------------------------------------------
# Input models: prompts
# ---------------------------------------------------------------------------

class Prompt(BaseModel):
    """A single natural-language prompt to be resolved.

    Attributes:
        prompt: The raw user question or instruction.
    """

    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_be_non_empty(cls, v: str) -> str:
        """Validate that the prompt string is not blank."""
        if not v.strip():
            raise ValueError("Prompt must not be empty")
        return v


# ---------------------------------------------------------------------------
# Output models: resolved function calls
# ---------------------------------------------------------------------------

class FunctionCall(BaseModel):
    """A resolved function call produced by constrained decoding.

    Attributes:
        prompt: The original natural-language request.
        name: Name of the function to call.
        parameters: Argument values keyed by parameter name.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]

    @model_validator(mode="after")
    def validate_parameters_not_none(self) -> "FunctionCall":
        """Ensure parameters dict exists (may be empty for zero-arg functions)."""
        if self.parameters is None:
            self.parameters = {}
        return self


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    """Runtime configuration parsed from CLI arguments.

    Attributes:
        functions_definition: Path to the function definitions JSON file.
        input_path: Path to the prompts JSON file.
        output_path: Path to write the results JSON file.
    """

    functions_definition: str = "data/input/functions_definition.json"
    input_path: str = "data/input/function_calling_tests.json"
    output_path: str = "data/output/function_calling_results.json"
    model_name: Optional[str] = None
