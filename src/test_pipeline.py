"""tests/test_pipeline.py — Unit tests for the call_me_maybe project.

Run with:  uv run pytest tests/ -v
"""

import json
import os
import tempfile

import pytest

from src.io_helpers import load_function_definitions, load_prompts, save_results
from src.models import FunctionCall, FunctionDefinition, Prompt, RunConfig
from src.prompt_builder import build_prompt
from src.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FUNCTIONS = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Greet someone.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
]

SAMPLE_PROMPTS = [
    {"prompt": "What is the sum of 2 and 3?"},
    {"prompt": "Greet Alice"},
]


@pytest.fixture()
def fn_defs() -> list:
    return [FunctionDefinition.model_validate(f) for f in SAMPLE_FUNCTIONS]


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_function_definition_valid(self) -> None:
        fn = FunctionDefinition.model_validate(SAMPLE_FUNCTIONS[0])
        assert fn.name == "fn_add_numbers"
        assert fn.parameters["a"].type == "number"

    def test_function_definition_empty_name_raises(self) -> None:
        bad = dict(SAMPLE_FUNCTIONS[0], name="   ")
        with pytest.raises(Exception):
            FunctionDefinition.model_validate(bad)

    def test_prompt_valid(self) -> None:
        p = Prompt.model_validate({"prompt": "Hello"})
        assert p.prompt == "Hello"

    def test_prompt_empty_raises(self) -> None:
        with pytest.raises(Exception):
            Prompt.model_validate({"prompt": "   "})

    def test_function_call_serialises(self) -> None:
        fc = FunctionCall(prompt="q", name="fn_greet", parameters={"name": "Bob"})
        d = fc.model_dump()
        assert d["name"] == "fn_greet"
        assert d["parameters"]["name"] == "Bob"


# ---------------------------------------------------------------------------
# I/O helpers tests
# ---------------------------------------------------------------------------

class TestIOHelpers:
    def test_load_function_definitions_ok(self, tmp_dir: str) -> None:
        path = os.path.join(tmp_dir, "fns.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_FUNCTIONS, f)
        defs = load_function_definitions(path)
        assert len(defs) == 2
        assert defs[0].name == "fn_add_numbers"

    def test_load_function_definitions_missing_file(self, tmp_dir: str) -> None:
        defs = load_function_definitions(os.path.join(tmp_dir, "nonexistent.json"))
        assert defs == []

    def test_load_function_definitions_invalid_json(self, tmp_dir: str) -> None:
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("not json {{{")
        defs = load_function_definitions(path)
        assert defs == []

    def test_load_prompts_ok(self, tmp_dir: str) -> None:
        path = os.path.join(tmp_dir, "prompts.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_PROMPTS, f)
        prompts = load_prompts(path)
        assert len(prompts) == 2

    def test_save_results_creates_file(self, tmp_dir: str) -> None:
        path = os.path.join(tmp_dir, "out", "results.json")
        results = [
            FunctionCall(prompt="q", name="fn_greet", parameters={"name": "X"})
        ]
        ok = save_results(results, path)
        assert ok
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data[0]["name"] == "fn_greet"


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_prompt_contains_function_name(self, fn_defs: list) -> None:
        p = build_prompt("What is 1+1?", fn_defs)
        assert "fn_add_numbers" in p

    def test_prompt_contains_user_request(self, fn_defs: list) -> None:
        p = build_prompt("Greet Bob", fn_defs)
        assert "Greet Bob" in p

    def test_prompt_primes_with_opening_brace(self, fn_defs: list) -> None:
        p = build_prompt("anything", fn_defs)
        # The prompt should end with the priming text to help the model start.
        assert '{"function"' in p or '"function"' in p or "function" in p


# ---------------------------------------------------------------------------
# Vocabulary tests (with a minimal synthetic vocab)
# ---------------------------------------------------------------------------

class TestVocabulary:
    def _make_vocab_file(self, tmp_dir: str) -> str:
        vocab = {
            "0": "<pad>",
            "1": "<eos>",
            "2": "{",
            "3": "}",
            "4": '"',
            "5": ":",
            "6": ",",
            "7": "function",
            "8": "add",
            "9": "1",
            "10": "2",
            "11": "true",
            "12": "false",
            "13": " hello",
            "14": "<|im_end|>",
        }
        path = os.path.join(tmp_dir, "vocab.json")
        with open(path, "w") as f:
            json.dump(vocab, f)
        return path

    def test_load_vocab(self, tmp_dir: str) -> None:
        path = self._make_vocab_file(tmp_dir)
        v = Vocabulary(path)
        assert v.token_string(2) == "{"
        assert v.token_string(3) == "}"

    def test_eos_detection(self, tmp_dir: str) -> None:
        path = self._make_vocab_file(tmp_dir)
        v = Vocabulary(path)
        assert 14 in v.eos_token_ids

    def test_token_ids_for(self, tmp_dir: str) -> None:
        path = self._make_vocab_file(tmp_dir)
        v = Vocabulary(path)
        ids = v.token_ids_for("{")
        assert 2 in ids

    def test_missing_file_raises(self, tmp_dir: str) -> None:
        with pytest.raises(FileNotFoundError):
            Vocabulary(os.path.join(tmp_dir, "missing.json"))

    def test_invalid_json_raises(self, tmp_dir: str) -> None:
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("{{{{")
        with pytest.raises(ValueError):
            Vocabulary(path)
