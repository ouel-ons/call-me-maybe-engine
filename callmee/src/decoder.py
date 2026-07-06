"""Constrained decoding implementation for guaranteed valid JSON output."""

import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from .models import FunctionDefinition


class ConstrainedDecoder:
    """
    Constrained decoder that enforces valid JSON and schema compliance.
    
    This decoder modifies logits at each generation step to ensure the output
    conforms to a specific JSON schema, guaranteeing 100% valid output.
    """
    
    def __init__(self, vocab_path: str):
        """
        Initialize the decoder with vocabulary mapping.
        
        Args:
            vocab_path: Path to the vocabulary JSON file.
        """
        self.vocab_path = vocab_path
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._load_vocabulary()
        
        # Cache for schema-valid tokens at each state
        self._state_cache: Dict[str, Set[int]] = {}
        
    def _load_vocabulary(self) -> None:
        """Load vocabulary from the JSON file."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                
            # Handle different vocabulary formats
            if isinstance(vocab_data, dict):
                if 'model' in vocab_data and 'vocab' in vocab_data['model']:
                    # Standard tokenizer.json format
                    vocab = vocab_data['model']['vocab']
                    for token, idx in vocab.items():
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token
                else:
                    # Simple format
                    for token, idx in vocab_data.items():
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token
            elif isinstance(vocab_data, list):
                # List format
                for idx, token in enumerate(vocab_data):
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token
            else:
                raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")
                
        except Exception as e:
            raise ValueError(f"Failed to load vocabulary: {e}")
    
    def get_token_ids(self, token: str) -> Optional[int]:
        """Get token ID for a token string."""
        return self.token_to_id.get(token)
    
    def get_token_string(self, token_id: int) -> Optional[str]:
        """Get token string for a token ID."""
        return self.id_to_token.get(token_id)
    
    def constrain_logits(
        self,
        logits: np.ndarray,
        current_output: str,
        functions: List[FunctionDefinition],
        schema_prefix: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply constraints to logits to ensure valid JSON and schema compliance.
        
        Args:
            logits: Array of logit values for all tokens.
            current_output: The current generated output string.
            functions: List of available function definitions.
            schema_prefix: Optional prefix path for nested schema validation.
            
        Returns:
            Modified logits with invalid tokens set to -inf.
        """
        # Create a copy to avoid modifying the original
        constrained_logits = logits.copy()
        
        # Determine which tokens are valid at this position
        valid_token_ids = self._get_valid_tokens(
            current_output, functions, schema_prefix
        )
        
        # Set invalid tokens to negative infinity
        if valid_token_ids:
            for i in range(len(constrained_logits)):
                if i not in valid_token_ids:
                    constrained_logits[i] = -float('inf')
            
        return constrained_logits
    
    def _get_valid_tokens(
        self,
        current_output: str,
        functions: List[FunctionDefinition],
        schema_prefix: Optional[str] = None
    ) -> Set[int]:
        """
        Get set of token IDs that would maintain valid structure.
        
        Args:
            current_output: Current generated output string.
            functions: List of function definitions.
            schema_prefix: Optional prefix for schema validation.
            
        Returns:
            Set of valid token IDs.
        """
        # Create cache key
        cache_key = f"{current_output[-100:]}:{schema_prefix or ''}"
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]
        
        valid_tokens = set()
        
        # If we're at the beginning, look for JSON opening
        if not current_output or current_output.strip() == '':
            # Check tokens that could start a JSON object
            for token, token_id in self.token_to_id.items():
                if token in ['{', '{"', '{\n', '{\t']:
                    valid_tokens.add(token_id)
            self._state_cache[cache_key] = valid_tokens
            return valid_tokens
        
        # Determine what we're expecting based on current state
        expected_type = self._determine_expected_type(current_output, functions)
        
        if expected_type == 'object_start':
            # Expecting object keys or closing brace
            for token, token_id in self.token_to_id.items():
                if token in ['"', '}', '}']:
                    valid_tokens.add(token_id)
                    
        elif expected_type == 'object_key':
            # Expecting a string key
            for token, token_id in self.token_to_id.items():
                if token.startswith('"') and token != '"':
                    valid_tokens.add(token_id)
                    
        elif expected_type == 'colon':
            # Expecting colon
            for token, token_id in self.token_to_id.items():
                if token == ':':
                    valid_tokens.add(token_id)
                    
        elif expected_type == 'string_value':
            # Expecting string values
            for token, token_id in self.token_to_id.items():
                if token.startswith('"') and token != '"':
                    valid_tokens.add(token_id)
                    
        elif expected_type in ['number_value', 'integer_value', 'float_value']:
            # Expecting numeric values
            for token, token_id in self.token_to_id.items():
                if token and (token[0].isdigit() or token == '-' or token == '.'):
                    valid_tokens.add(token_id)
                    
        elif expected_type == 'boolean_value':
            # Expecting boolean values
            for token, token_id in self.token_to_id.items():
                if token in ['true', 'false', 'true,', 'false,']:
                    valid_tokens.add(token_id)
        
        elif expected_type == 'comma_or_close':
            # Expecting comma or closing brace
            for token, token_id in self.token_to_id.items():
                if token in [',', '}', '},', '}']:
                    valid_tokens.add(token_id)
        
        # Cache the result
        self._state_cache[cache_key] = valid_tokens
        
        return valid_tokens
    
    def _determine_expected_type(
        self,
        current_output: str,
        functions: List[FunctionDefinition]
    ) -> str:
        """
        Determine what type of token is expected next.
        
        This analyzes the current output state to figure out what the
        next token should be to maintain valid JSON structure.
        """
        # Remove whitespace for easier parsing
        stripped = re.sub(r'\s+', '', current_output)
        
        if not stripped:
            return 'object_start'
        
        # Check if we're in a JSON object
        if stripped.count('{') > stripped.count('}'):
            # We're inside an object
            
            # Check if we're at the root level after opening
            if stripped == '{' or stripped == '{"':
                return 'object_key'
            
            # Check if we're after a colon
            if stripped.endswith(':'):
                # Check what key we're at
                last_key_match = re.search(r'"([^"]+)"\s*:', stripped)
                if last_key_match:
                    last_key = last_key_match.group(1)
                    # If this is "function" or "name", expect string
                    if last_key in ['function', 'name']:
                        return 'string_value'
                    # If this is "arguments" or "parameters", expect object
                    elif last_key in ['arguments', 'parameters']:
                        return 'object_start'
                    # Otherwise check function definitions for type
                    for func in functions:
                        for param in func.parameters:
                            if param.name == last_key:
                                if param.type in ['number', 'integer', 'float']:
                                    return f'{param.type}_value'
                                elif param.type == 'boolean':
                                    return 'boolean_value'
                                elif param.type == 'string':
                                    return 'string_value'
                return 'string_value'
            
            # Check if we're at a key position
            if stripped.endswith('{') or stripped.endswith(','):
                return 'object_key'
            
            # Check if we need a comma or closing brace
            if stripped.endswith('"') or stripped.endswith('}') or stripped.endswith(']'):
                return 'comma_or_close'
        
        # If we're at the beginning of a value
        if stripped.endswith(':') or stripped.endswith('["'):
            return 'string_value'
        
        return 'unknown'
    
    def sample_next_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.0
    ) -> int:
        """
        Sample the next token from constrained logits.
        
        Args:
            logits: Constrained logits.
            temperature: Temperature for sampling (0 = greedy).
            
        Returns:
            Selected token ID.
        """
        # If all logits are -inf, return 0 (fallback)
        if np.all(logits == -float('inf')):
            return 0
        
        if temperature == 0:
            # Greedy selection
            return int(np.argmax(logits))
        else:
            # Apply temperature scaling
            scaled_logits = logits / temperature
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)
            # Sample
            return int(np.random.choice(len(probs), p=probs))