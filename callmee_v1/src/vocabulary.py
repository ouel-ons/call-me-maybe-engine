"""vocabulary.py — Load and index the model vocabulary.

The llm_sdk exposes get_path_to_vocabulary_json() which returns a path to a
JSON file mapping token-id (as string) → token-string. We load this once and
build fast lookup structures used by the constrained decoder.
"""

import json
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Vocabulary:
    """Indexed model vocabulary for fast token lookups.

    Attributes:
        id_to_token: Mapping from integer token ID to token string.
        token_to_ids: Mapping from token string to list of token IDs
                      (multiple IDs can share the same surface form).
        eos_token_ids: Set of end-of-sequence token IDs.
    """

    def __init__(self, vocab_path: str) -> None:
        """Load vocabulary from the JSON file produced by the SDK.

        Args:
            vocab_path: Absolute path to the vocabulary JSON file.

        Raises:
            FileNotFoundError: If the vocab file does not exist.
            ValueError: If the file contains invalid JSON or unexpected format.
        """
        self.id_to_token: Dict[int, str] = {}
        self.token_to_ids: Dict[str, List[int]] = {}
        self.eos_token_ids: Set[int] = set()

        self._load(vocab_path)
        self._index_eos()

    def _load(self, path: str) -> None:
        """Read and parse the vocabulary JSON file.

        Args:
            path: Path to the vocabulary JSON.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw: Dict[str, str] = json.load(fh)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found: {path}"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Vocabulary file contains invalid JSON: {exc}"
            )

        for id_str, token_str in raw.items():
            try:
                token_id = int(id_str)
            except ValueError:
                logger.warning("Skipping non-integer token ID key: %s", id_str)
                continue

            self.id_to_token[token_id] = token_str

            if token_str not in self.token_to_ids:
                self.token_to_ids[token_str] = []
            self.token_to_ids[token_str].append(token_id)

        logger.info("Loaded vocabulary with %d tokens", len(self.id_to_token))

    def _index_eos(self) -> None:
        """Identify end-of-sequence tokens by common surface forms."""
        eos_forms = {"<|endoftext|>", "</s>", "<eos>", "<|im_end|>", "<|end|>"}
        for form in eos_forms:
            if form in self.token_to_ids:
                for tid in self.token_to_ids[form]:
                    self.eos_token_ids.add(tid)

    def token_string(self, token_id: int) -> Optional[str]:
        """Return the surface string for a token ID, or None if unknown.

        Args:
            token_id: Integer token identifier.

        Returns:
            Token string or None.
        """
        return self.id_to_token.get(token_id)

    def token_ids_for(self, surface: str) -> List[int]:
        """Return all token IDs that map to a given surface form.

        Args:
            surface: Token string (exact match, including leading spaces).

        Returns:
            Possibly-empty list of integer token IDs.
        """
        return self.token_to_ids.get(surface, [])

    def ids_that_start_with(self, prefix: str) -> List[int]:
        """Return token IDs whose surface form starts with *prefix*.

        This is used to find all tokens valid at a position where we have
        already committed to a partial string (e.g., the opening characters
        of a JSON string value).

        Args:
            prefix: The prefix to match against.

        Returns:
            List of matching token IDs.
        """
        result: List[int] = []
        for surface, ids in self.token_to_ids.items():
            if surface.startswith(prefix) or prefix.startswith(surface):
                result.extend(ids)
        return result

    def ids_that_contain(self, substring: str) -> List[int]:
        """Return token IDs whose surface form contains *substring*.

        Args:
            substring: The substring to search for.

        Returns:
            List of matching token IDs.
        """
        result: List[int] = []
        for surface, ids in self.token_to_ids.items():
            if substring in surface:
                result.extend(ids)
        return result
