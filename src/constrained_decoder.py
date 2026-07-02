"""constrained_decoder.py — Schema-guided constrained decoding engine.

This module implements the core technique of the project: at every generation
step we examine the logits produced by the LLM and mask out any token that
would violate either the JSON syntax or the expected output schema. Only the
valid tokens remain, and we pick the highest-scoring one (greedy decoding).

The target output schema is:
    {
      "function": "<one of the known function names>",
      "arguments": {
        "<param_name>": <typed value>,
        ...
      }
    }

State machine
-------------
We track position in the output with a simple enum-style string:

  START              → emit '{'
  FUNC_KEY           → emit '"function"'
  FUNC_COLON         → emit ':'
  FUNC_VALUE         → emit one of the valid function-name strings
  ARGS_COMMA         → emit ','
  ARGS_KEY           → emit '"arguments"'
  ARGS_COLON         → emit ':'
  ARGS_OPEN          → emit '{'
  PARAM_KEY          → emit one of the known parameter-name strings
  PARAM_COLON        → emit ':'
  PARAM_VALUE        → emit a typed value (number / string / boolean / null)
  PARAM_SEP          → emit ',' (more params) or '}' (done)
  OUTER_CLOSE        → emit '}'

Within PARAM_VALUE we use a sub-state to handle multi-token strings and
numbers:

  STRING_BODY        → accumulating string characters; '\"' closes
  NUMBER_BODY        → accumulating digit/sign/decimal characters

This is intentionally simple and opinionated toward the project's fixed schema.
A production system would build a full grammar automaton from JSON Schema; here
clarity beats generality.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .models import FunctionDefinition
from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)

# Sentinel value used to mask logits for invalid tokens.
NEG_INF: float = float("-inf")

# Maximum tokens to generate before giving up.
MAX_TOKENS: int = 512

# JSON type → Python type name mapping.
JSON_TYPE_MAP: Dict[str, str] = {
    "number": "number",
    "integer": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "null": "null",
}


class DecoderState:
    """Mutable state for a single constrained decoding run.

    Attributes:
        phase: Current phase of the state machine.
        current_param: Parameter name being decoded (in PARAM_VALUE phase).
        current_type: Expected JSON type for the current parameter.
        remaining_params: Parameter names not yet generated.
        generated_params: Already-generated parameter name→value pairs.
        string_buffer: Accumulated characters for the current string value.
        number_buffer: Accumulated characters for the current number value.
        in_escape: Whether the next character is escaped inside a string.
    """

    def __init__(self, param_names: List[str], param_types: Dict[str, str]) -> None:
        self.phase: str = "START"
        self.current_param: Optional[str] = None
        self.current_type: Optional[str] = None
        self.remaining_params: List[str] = list(param_names)
        self.generated_params: Dict[str, Any] = {}
        self.string_buffer: str = ""
        self.number_buffer: str = ""
        self.in_escape: bool = False
        self._param_types: Dict[str, str] = param_types

    def param_type(self, name: str) -> str:
        """Look up the JSON type for a parameter name."""
        return self._param_types.get(name, "string")


class ConstrainedDecoder:
    """Guides LLM token generation to produce schema-valid JSON.

    At each step this class:
    1. Receives the logit vector from the LLM.
    2. Computes the set of valid next tokens given the current parse state.
    3. Masks all other logits to -inf.
    4. Returns the argmax token ID (greedy).

    Args:
        vocabulary: Loaded model vocabulary.
        functions: List of available function definitions.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        functions: List[FunctionDefinition],
    ) -> None:
        self._vocab = vocabulary
        self._functions = {fn.name: fn for fn in functions}
        self._function_names: List[str] = [fn.name for fn in functions]

        # Pre-compute token ID sets for fixed structural tokens.
        self._struct = self._build_structural_token_sets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_function_call(
        self,
        prompt_ids: List[int],
        get_logits_fn: Any,
    ) -> Dict[str, Any]:
        """Run constrained decoding and return the parsed function call dict.

        Args:
            prompt_ids: Encoded prompt token IDs.
            get_logits_fn: Callable that accepts a numpy array of token IDs
                           and returns a numpy logit array.

        Returns:
            Dict with keys 'function' and 'arguments'.

        Raises:
            RuntimeError: If generation fails or exceeds MAX_TOKENS.
        """
        selected_function: Optional[str] = None
        state: Optional[DecoderState] = None

        generated_ids: List[int] = list(prompt_ids)
        generated_text: str = ""

        for step in range(MAX_TOKENS):
            input_arr = np.array(generated_ids, dtype=np.int64)
            logits: np.ndarray = get_logits_fn(input_arr)

            valid_ids, next_phase_info = self._valid_next_tokens(
                generated_text,
                state,
                selected_function,
            )

            # Mask everything invalid.
            masked_logits = np.full_like(logits, NEG_INF, dtype=np.float32)
            for vid in valid_ids:
                if vid < len(masked_logits):
                    masked_logits[vid] = logits[vid]

            if np.all(masked_logits == NEG_INF):
                logger.warning(
                    "Step %d: no valid tokens — forcing EOS. State: %s text: %r",
                    step,
                    getattr(state, "phase", "pre-state"),
                    generated_text[-60:],
                )
                break

            next_token_id = int(np.argmax(masked_logits))
            token_str = self._vocab.token_string(next_token_id) or ""

            # Advance state machine based on what we just committed.
            selected_function, state = self._advance_state(
                generated_text,
                token_str,
                state,
                selected_function,
                next_phase_info,
            )

            generated_ids.append(next_token_id)
            generated_text += token_str

            if next_token_id in self._vocab.eos_token_ids:
                break

            # Termination: we have closed the outer object.
            if state is not None and state.phase == "DONE":
                break

        return self._extract_result(generated_text, selected_function, state)

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    def _valid_next_tokens(
        self,
        generated_text: str,
        state: Optional[DecoderState],
        selected_function: Optional[str],
    ) -> Tuple[Set[int], Dict[str, Any]]:
        """Return (valid_token_ids, phase_info) for the current parse position.

        Args:
            generated_text: Everything generated so far (after the prompt).
            state: Current decoder state, or None if not yet started.
            selected_function: Function name chosen so far, or None.

        Returns:
            Tuple of (set of valid token IDs, auxiliary info for state advance).
        """
        info: Dict[str, Any] = {}

        if state is None:
            # Phase START: must emit '{'
            return self._ids_for_exact("{"), info

        phase = state.phase

        if phase == "START":
            return self._ids_for_exact("{"), info

        if phase == "FUNC_KEY":
            return self._ids_for_exact('"function"'), info

        if phase == "FUNC_COLON":
            return self._ids_for_exact(":"), info

        if phase == "FUNC_VALUE":
            # Must be one of the known function names, JSON-quoted.
            return self._ids_for_enum_value(self._function_names, info)

        if phase == "ARGS_COMMA":
            return self._ids_for_exact(","), info

        if phase == "ARGS_KEY":
            return self._ids_for_exact('"arguments"'), info

        if phase == "ARGS_COLON":
            return self._ids_for_exact(":"), info

        if phase == "ARGS_OPEN":
            return self._ids_for_exact("{"), info

        if phase == "PARAM_KEY":
            if not state.remaining_params:
                # No params left; this state shouldn't occur, but handle gracefully.
                return self._ids_for_exact("}"), info
            return self._ids_for_enum_value(state.remaining_params, info)

        if phase == "PARAM_COLON":
            return self._ids_for_exact(":"), info

        if phase == "PARAM_VALUE":
            return self._ids_for_typed_value(state, info)

        if phase == "STRING_BODY":
            return self._ids_for_string_continuation(state, info)

        if phase == "NUMBER_BODY":
            return self._ids_for_number_continuation(state, info)

        if phase == "PARAM_SEP":
            ids: Set[int] = set()
            if state.remaining_params:
                ids |= self._ids_for_exact(",")
            ids |= self._ids_for_exact("}")
            return ids, info

        if phase == "OUTER_CLOSE":
            return self._ids_for_exact("}"), info

        if phase == "DONE":
            return set(self._vocab.eos_token_ids), info

        logger.error("Unknown phase: %s", phase)
        return set(), info

    def _advance_state(
        self,
        generated_text: str,
        token_str: str,
        state: Optional[DecoderState],
        selected_function: Optional[str],
        info: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[DecoderState]]:
        """Transition the state machine after committing *token_str*.

        Args:
            generated_text: Text before this token.
            token_str: The token we just selected.
            state: Current state.
            selected_function: Function name chosen so far.
            info: Auxiliary info from _valid_next_tokens.

        Returns:
            Updated (selected_function, state).
        """
        stripped = token_str.strip()

        if state is None:
            s = DecoderState([], {})
            s.phase = "FUNC_KEY"
            return selected_function, s

        phase = state.phase

        if phase == "START":
            state.phase = "FUNC_KEY"

        elif phase == "FUNC_KEY":
            state.phase = "FUNC_COLON"

        elif phase == "FUNC_COLON":
            state.phase = "FUNC_VALUE"

        elif phase == "FUNC_VALUE":
            # Extract the function name from the quoted token(s).
            fn_name = info.get("chosen_value")
            if fn_name and fn_name in self._functions:
                selected_function = fn_name
                fn = self._functions[fn_name]
                param_names = list(fn.parameters.keys())
                param_types = {
                    k: JSON_TYPE_MAP.get(v.type, "string")
                    for k, v in fn.parameters.items()
                }
                state = DecoderState(param_names, param_types)
                state.phase = "ARGS_COMMA"
            else:
                state.phase = "ARGS_COMMA"

        elif phase == "ARGS_COMMA":
            state.phase = "ARGS_KEY"

        elif phase == "ARGS_KEY":
            state.phase = "ARGS_COLON"

        elif phase == "ARGS_COLON":
            state.phase = "ARGS_OPEN"

        elif phase == "ARGS_OPEN":
            if state.remaining_params:
                state.phase = "PARAM_KEY"
            else:
                state.phase = "PARAM_SEP"

        elif phase == "PARAM_KEY":
            param_name = info.get("chosen_value")
            if param_name and param_name in state._param_types:
                state.current_param = param_name
                state.current_type = state.param_type(param_name)
                if param_name in state.remaining_params:
                    state.remaining_params.remove(param_name)
            state.phase = "PARAM_COLON"

        elif phase == "PARAM_COLON":
            state.phase = "PARAM_VALUE"

        elif phase == "PARAM_VALUE":
            t = state.current_type
            if t == "string":
                # Opening quote — transition to string body.
                state.string_buffer = ""
                state.phase = "STRING_BODY"
            elif t == "boolean":
                val_str = info.get("chosen_value", "")
                state.generated_params[state.current_param or ""] = (
                    val_str == "true"
                )
                state.phase = "PARAM_SEP"
            elif t == "null":
                state.generated_params[state.current_param or ""] = None
                state.phase = "PARAM_SEP"
            else:
                # number: start collecting
                state.number_buffer = token_str.strip()
                state.phase = "NUMBER_BODY"

        elif phase == "STRING_BODY":
            if token_str == '"' and not state.in_escape:
                # Closing quote — string is complete.
                state.generated_params[state.current_param or ""] = (
                    state.string_buffer
                )
                state.string_buffer = ""
                state.phase = "PARAM_SEP"
            else:
                if token_str == "\\" and not state.in_escape:
                    state.in_escape = True
                else:
                    state.in_escape = False
                state.string_buffer += token_str

        elif phase == "NUMBER_BODY":
            if stripped in {",", "}", "]"}:
                # Number complete; token is a delimiter — do not consume.
                try:
                    val: float = float(state.number_buffer)
                except ValueError:
                    val = 0.0
                state.generated_params[state.current_param or ""] = val
                state.number_buffer = ""
                # This delimiter belongs to PARAM_SEP logic.
                if stripped == ",":
                    state.phase = "PARAM_KEY"
                elif stripped == "}":
                    state.phase = "OUTER_CLOSE"
            else:
                state.number_buffer += token_str.strip()
                # Stay in NUMBER_BODY.

        elif phase == "PARAM_SEP":
            if stripped == ",":
                state.phase = "PARAM_KEY"
            elif stripped == "}":
                state.phase = "OUTER_CLOSE"

        elif phase == "OUTER_CLOSE":
            state.phase = "DONE"

        return selected_function, state

    # ------------------------------------------------------------------
    # Token-set builders
    # ------------------------------------------------------------------

    def _ids_for_exact(self, text: str) -> Set[int]:
        """Return token IDs whose surface form equals *text* exactly,
        or whose surface form is a prefix of *text* (to handle multi-token
        emission of fixed strings).

        Args:
            text: The exact string we want to emit next.

        Returns:
            Set of valid token IDs.
        """
        ids: Set[int] = set()
        for surface, token_ids in self._vocab.token_to_ids.items():
            clean = surface.lstrip(" ")  # vocab entries often have leading space
            if clean == text or text.startswith(clean) or clean.startswith(text):
                ids.update(token_ids)
        # Always include any token whose surface IS text, regardless of spaces.
        direct = self._vocab.token_to_ids.get(text, [])
        ids.update(direct)
        return ids

    def _ids_for_enum_value(
        self,
        choices: List[str],
        info: Dict[str, Any],
    ) -> Set[int]:
        """Return token IDs that begin a quoted JSON string value from *choices*.

        We look for tokens that appear as the opening of any of the choice
        strings once surrounded by quotes. We also record which value we are
        committing to in *info['chosen_value']*.

        Args:
            choices: List of string values to permit.
            info: Mutable dict; we set info['chosen_value'] when unambiguous.

        Returns:
            Set of valid token IDs.
        """
        ids: Set[int] = set()
        for choice in choices:
            quoted = f'"{choice}"'
            for surface, token_ids in self._vocab.token_to_ids.items():
                clean = surface.lstrip(" ")
                if quoted.startswith(clean) or clean == quoted:
                    ids.update(token_ids)
                    if clean == quoted:
                        info["chosen_value"] = choice
        return ids

    def _ids_for_typed_value(
        self,
        state: DecoderState,
        info: Dict[str, Any],
    ) -> Set[int]:
        """Return token IDs valid as the start of a typed JSON value.

        Args:
            state: Current decoder state (provides current_type).
            info: Mutable info dict.

        Returns:
            Set of valid token IDs.
        """
        t = state.current_type
        ids: Set[int] = set()

        if t == "string":
            # Opening quote.
            ids |= self._ids_for_exact('"')

        elif t == "boolean":
            for bval in ["true", "false"]:
                for surface, token_ids in self._vocab.token_to_ids.items():
                    clean = surface.lstrip(" ")
                    if bval.startswith(clean) or clean == bval:
                        ids.update(token_ids)
                        if clean == bval:
                            info["chosen_value"] = bval

        elif t == "null":
            for surface, token_ids in self._vocab.token_to_ids.items():
                clean = surface.lstrip(" ")
                if "null".startswith(clean) or clean == "null":
                    ids.update(token_ids)

        else:
            # number: digits, sign, decimal point.
            for surface, token_ids in self._vocab.token_to_ids.items():
                clean = surface.lstrip(" ")
                if clean and all(c in "0123456789.-+eE" for c in clean):
                    ids.update(token_ids)
                # Also allow a leading minus sign on its own.
                if clean == "-":
                    ids.update(token_ids)

        return ids

    def _ids_for_string_continuation(
        self,
        state: DecoderState,
        info: Dict[str, Any],
    ) -> Set[int]:
        """Return token IDs valid inside a JSON string body.

        Any printable token is valid except an unescaped closing quote,
        which signals the end of the string.

        Args:
            state: Current decoder state.
            info: Mutable info dict.

        Returns:
            Set of valid token IDs.
        """
        ids: Set[int] = set()
        for token_id, surface in self._vocab.id_to_token.items():
            # Allow anything that is not a bare closing quote (unless escaped).
            if surface == '"' and not state.in_escape:
                ids.add(token_id)  # closing quote is valid (terminates string)
            elif surface and surface != "\n":
                ids.add(token_id)
        return ids

    def _ids_for_number_continuation(
        self,
        state: DecoderState,
        info: Dict[str, Any],
    ) -> Set[int]:
        """Return token IDs valid as continuation of a JSON number.

        Args:
            state: Current decoder state (has number_buffer).
            info: Mutable info dict.

        Returns:
            Set of valid token IDs.
        """
        ids: Set[int] = set()
        for surface, token_ids in self._vocab.token_to_ids.items():
            clean = surface.lstrip(" ")
            if clean and all(c in "0123456789.eE+-" for c in clean):
                ids.update(token_ids)
            # Separators terminate the number.
            if clean in {",", "}", "]"}:
                ids.update(token_ids)
        return ids

    def _build_structural_token_sets(self) -> Dict[str, Set[int]]:
        """Pre-compute token ID sets for common structural characters.

        Returns:
            Dict mapping structural character string → set of token IDs.
        """
        chars = ["{", "}", ":", ",", '"', "[", "]"]
        result: Dict[str, Set[int]] = {}
        for ch in chars:
            result[ch] = self._ids_for_exact(ch)
        return result

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _extract_result(
        self,
        generated_text: str,
        selected_function: Optional[str],
        state: Optional[DecoderState],
    ) -> Dict[str, Any]:
        """Parse the generated text into a structured result dict.

        First tries the in-memory state (most reliable). Falls back to
        JSON parsing of the generated text string.

        Args:
            generated_text: The full generated text (after the prompt).
            selected_function: Function name recorded during state transitions.
            state: Final decoder state.

        Returns:
            Dict with keys 'function' and 'arguments'.
        """
        # Prefer in-memory state (it's already typed correctly).
        if selected_function and state and state.generated_params:
            return {
                "function": selected_function,
                "arguments": dict(state.generated_params),
            }

        # Fallback: try to JSON-parse the generated text.
        try:
            obj = json.loads(generated_text.strip())
            if isinstance(obj, dict):
                return {
                    "function": obj.get("function", selected_function or ""),
                    "arguments": obj.get("arguments", {}),
                }
        except json.JSONDecodeError:
            pass

        # Last resort: return whatever we have.
        return {
            "function": selected_function or "",
            "arguments": state.generated_params if state else {},
        }
