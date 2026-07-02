*This project has been created as part of the 42 curriculum.*

# call me maybe — LLM Function Calling with Constrained Decoding

## Description

This project implements a **function-calling tool** that translates natural-language
requests into structured JSON function calls. Given a prompt like *"What is the sum
of 40 and 2?"*, the system produces:

```json
{
  "function": "fn_add_numbers",
  "arguments": { "a": 40.0, "b": 2.0 }
}
```

The key insight is that small LLMs (≤ 1B parameters) are notoriously unreliable at
producing valid structured output when prompted alone — success rates as low as 30%.
This project achieves near-100% valid JSON by implementing **constrained decoding**:
at each generation step, we mask any token that would violate the JSON schema to
`-inf` before sampling.

## Algorithm Explanation

### Constrained Decoding

The LLM generates text one token at a time. At each step it produces a **logit vector**
of shape `[vocab_size]` — one raw score per token. Normally you sample from these
scores. Constrained decoding intercepts before sampling:

1. Obtain the logit vector from the LLM.
2. Determine which tokens are valid continuations given the current parse state and schema.
3. Set all other token logits to `−∞` (so their softmax probability = 0).
4. Pick the argmax of the remaining valid tokens (greedy decoding).

The **parse state** is a small state machine:

```
START → FUNC_KEY → FUNC_COLON → FUNC_VALUE → ARGS_COMMA → ARGS_KEY →
ARGS_COLON → ARGS_OPEN → PARAM_KEY → PARAM_COLON → PARAM_VALUE →
[STRING_BODY | NUMBER_BODY] → PARAM_SEP → ... → OUTER_CLOSE → DONE
```

At `FUNC_VALUE`, only tokens that spell out one of the known function names (quoted)
are allowed. At `PARAM_VALUE`, only tokens consistent with the declared parameter
type (number, string, boolean, null) are allowed.

### Vocabulary Indexing

The SDK provides a JSON file mapping token IDs to their string representations.
We load this once and build reverse indexes for fast lookup:
- `id → surface string`
- `surface string → list of IDs` (multiple IDs can share the same surface)

This lets us efficiently compute which token IDs correspond to structural characters
like `{`, `}`, `"`, `:`, `,` as well as which IDs begin a particular function name
or parameter type.

## Design Decisions

- **State machine over grammar automaton**: A full context-free grammar automaton
  (like Outlines uses) would be more general, but the project schema is fixed, making
  a hand-written state machine simpler to understand and debug.
- **Greedy decoding**: We use argmax rather than sampling because correctness is
  more important than diversity for structured output.
- **In-memory result vs JSON parse fallback**: The state machine records values as it
  generates them (correctly typed). We only JSON-parse the generated text string as
  a fallback, avoiding edge cases with JSON parsing of partially-valid outputs.
- **Pydantic v2** for all data models — validation at the boundary ensures that malformed
  input files produce clear error messages rather than cryptic KeyErrors downstream.

## Performance Analysis

- **JSON validity**: 100% — the constrained decoder guarantees this by construction.
- **Function selection accuracy**: Depends on the LLM's understanding of the prompt.
  With Qwen3-0.6B and the semantic prompt provided, accuracy on the sample test set
  should be ≥ 90%.
- **Speed**: Each prompt requires one forward pass per generated token. The output
  for the sample schema is typically 30–60 tokens, so 5 minutes is sufficient for
  dozens of prompts on standard CPU hardware.

## Challenges Faced

1. **Multi-token strings**: A function name like `fn_add_numbers` may be split across
   several tokens by the BPE tokenizer. The `_ids_for_enum_value` method handles this
   by finding all tokens that are prefixes or extensions of the quoted name.

2. **Leading-space tokens**: BPE tokenizers often encode a space as part of the token
   (e.g. `" hello"` vs `"hello"`). The decoder strips leading spaces when comparing
   surface forms to structural characters.

3. **Number termination**: A number like `42.0` has no explicit closing delimiter in
   the state machine — it terminates when the next separator (`}` or `,`) is seen.
   The `NUMBER_BODY` state handles this by treating separator tokens as terminators
   without consuming them.

## Testing Strategy

- **Unit tests** (`tests/test_pipeline.py`) cover: model validation, I/O error handling,
  prompt construction, and vocabulary loading — all with synthetic data so the tests
  run without the real LLM model.
- **Integration test**: Run the full pipeline with the sample input files and inspect
  `data/output/function_calling_results.json` for correctness.
- **Edge cases tested**: missing files, invalid JSON, empty prompt, empty function list,
  unknown parameter types.

## Instructions

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <your-repo-url>
cd call_me_maybe
# Copy the provided llm_sdk package into the project root.
cp -r /path/to/llm_sdk ./llm_sdk
uv sync
```

### Running

```bash
# Default paths
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Example Usage

```bash
$ uv run python -m src
12:34:01 [INFO] __main__: Starting call-me-maybe pipeline
12:34:01 [INFO] pipeline: Loaded 3 function definitions
12:34:01 [INFO] pipeline: Loaded 8 prompts
12:34:01 [INFO] pipeline: Processing prompt 1/8: 'What is the sum of 2 and 3?'
...
12:34:45 [INFO] pipeline: Wrote 8 results to data/output/function_calling_results.json
```

Output file:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  ...
]
```

### Testing

```bash
uv run pytest tests/ -v
```

### Linting

```bash
make lint
```

## Resources

### Core Concepts

- **BPE Tokenization**: Karpathy, "Let's build the GPT tokenizer" (YouTube, 2024)
- **LLM Generation**: Jay Alammar, "The Illustrated GPT-2" — jalammar.github.io
- **Constrained Decoding**: Willard & Louf, "Efficient Guided Generation for Large
  Language Models" (arXiv 2023, the Outlines paper)
- **JSON Schema**: "Understanding JSON Schema" — json-schema.org/learn
- **Pydantic v2**: docs.pydantic.dev/latest

### Python Tooling

- **mypy**: mypy.readthedocs.io — especially "Getting started" and type hints guide
- **flake8**: flake8.pycqa.org
- **uv**: docs.astral.sh/uv

### AI Usage

AI was used for:
- Drafting initial docstrings (reviewed and corrected manually)
- Suggesting edge cases for the test suite
- Explaining the Outlines paper's approach to context-free grammar automata

All generated code was reviewed, understood, and tested before inclusion.
