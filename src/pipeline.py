"""pipeline.py — Orchestrates the full function-calling pipeline.

For each prompt:
  1. Build a prompt string that presents the available functions.
  2. Tokenise the prompt.
  3. Run constrained decoding to generate a valid JSON function call.
  4. Parse the result into a FunctionCall object.
"""

import logging
from typing import List, Optional

from .constrained_decoder import ConstrainedDecoder
from .io_helpers import load_function_definitions, load_prompts, save_results
from .models import FunctionCall, RunConfig
from .prompt_builder import build_prompt
from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def run(config: RunConfig) -> List[FunctionCall]:
    """Execute the full pipeline and return the list of function calls.

    Args:
        config: Runtime configuration (paths, model name, etc.).

    Returns:
        List of FunctionCall objects, one per prompt.
    """
    # Load inputs.
    functions = load_function_definitions(config.functions_definition)
    if not functions:
        logger.error("No function definitions loaded — cannot proceed.")
        return []

    prompts = load_prompts(config.input_path)
    if not prompts:
        logger.warning("No prompts loaded — output will be empty.")

    # Initialise model.
    try:
        from llm_sdk import Small_LLM_Model  # type: ignore[import]
        model = Small_LLM_Model()
    except Exception as exc:
        logger.error("Failed to initialise Small_LLM_Model: %s", exc)
        return []

    # Load vocabulary.
    try:
        vocab_path: str = model.get_path_to_vocabulary_json()
        vocab = Vocabulary(vocab_path)
    except Exception as exc:
        logger.error("Failed to load vocabulary: %s", exc)
        return []

    # Build constrained decoder.
    decoder = ConstrainedDecoder(vocab, functions)

    results: List[FunctionCall] = []

    for i, prompt_obj in enumerate(prompts):
        logger.info(
            "Processing prompt %d/%d: %r", i + 1, len(prompts), prompt_obj.prompt
        )

        try:
            result = _process_single(
                prompt_obj.prompt,
                functions,
                model,
                decoder,
            )
            results.append(result)
        except Exception as exc:
            logger.error(
                "Failed to process prompt %d (%r): %s",
                i + 1,
                prompt_obj.prompt,
                exc,
            )
            # Add a placeholder so output indices stay aligned.
            results.append(
                FunctionCall(
                    prompt=prompt_obj.prompt,
                    name="",
                    parameters={},
                )
            )

    # Save output.
    save_results(results, config.output_path)

    return results


def _process_single(
    user_request: str,
    functions: list,
    model: object,
    decoder: ConstrainedDecoder,
) -> FunctionCall:
    """Process one prompt through the full pipeline.

    Args:
        user_request: The natural-language prompt string.
        functions: List of available function definitions.
        model: Initialised Small_LLM_Model instance.
        decoder: Configured ConstrainedDecoder.

    Returns:
        Validated FunctionCall object.
    """
    # Build and tokenise the prompt.
    prompt_text = build_prompt(user_request, functions)
    prompt_ids = model.encode(prompt_text)  # type: ignore[attr-defined]

    # Run constrained decoding.
    raw_result = decoder.decode_function_call(
        prompt_ids,
        get_logits_fn=model.get_logits_from_input_ids,  # type: ignore[attr-defined]
    )

    fn_name: str = raw_result.get("function", "")
    arguments: dict = raw_result.get("arguments", {})

    # Type-coerce arguments to match the function definition schema.
    fn_map = {f.name: f for f in functions}
    if fn_name in fn_map:
        fn_def = fn_map[fn_name]
        arguments = _coerce_arguments(arguments, fn_def)

    return FunctionCall(
        prompt=user_request,
        name=fn_name,
        parameters=arguments,
    )


def _coerce_arguments(arguments: dict, fn_def: object) -> dict:
    """Coerce argument values to their declared types.

    Args:
        arguments: Raw decoded argument dict (values may be strings).
        fn_def: FunctionDefinition with declared parameter types.

    Returns:
        Argument dict with values coerced to correct Python types.
    """
    coerced: dict = {}
    try:
        params = fn_def.parameters  # type: ignore[attr-defined]
        for param_name, param_type_obj in params.items():
            raw_val = arguments.get(param_name)
            type_str = param_type_obj.type.lower()

            if raw_val is None:
                coerced[param_name] = raw_val
                continue

            try:
                if type_str in {"number", "integer"}:
                    coerced[param_name] = float(str(raw_val))
                elif type_str == "boolean":
                    if isinstance(raw_val, bool):
                        coerced[param_name] = raw_val
                    else:
                        coerced[param_name] = str(raw_val).lower() == "true"
                elif type_str == "null":
                    coerced[param_name] = None
                else:
                    coerced[param_name] = str(raw_val)
            except (ValueError, TypeError):
                coerced[param_name] = raw_val

        # Preserve any extra keys not in the definition (shouldn't happen).
        for k, v in arguments.items():
            if k not in coerced:
                coerced[k] = v

    except AttributeError:
        return arguments

    return coerced
