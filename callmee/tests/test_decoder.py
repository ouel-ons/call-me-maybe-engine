
"""Unit tests for the constrained decoder."""

import unittest
import tempfile
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.decoder import ConstrainedDecoder
from src.models import FunctionDefinition, FunctionParameter


class TestConstrainedDecoder(unittest.TestCase):
    """Test cases for the ConstrainedDecoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary vocabulary file
        self.vocab_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.vocab_data = {
            "": 0, " ": 1, "a": 2, "b": 3, "c": 4,
            "{": 5, "}": 6, '"': 7, ":": 8, ",": 9,
            "function": 10, "arguments": 11, "name": 12,
            "true": 13, "false": 14,
            "1": 16, "2": 17, "3": 18, "4": 19, "5": 20,
            "-": 26, ".": 27
        }
        json.dump(self.vocab_data, self.vocab_file)
        self.vocab_file.close()
        
        self.functions = [
            FunctionDefinition(
                name="add_numbers",
                description="Add two numbers",
                parameters=[
                    FunctionParameter(name="a", type="number", description="First number"),
                    FunctionParameter(name="b", type="number", description="Second number")
                ],
                returns="number"
            )
        ]
        
        self.decoder = ConstrainedDecoder(self.vocab_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.vocab_file.name)
    
    def test_vocabulary_loading(self):
        """Test vocabulary loading."""
        self.assertIsNotNone(self.decoder.token_to_id)
        self.assertIsNotNone(self.decoder.id_to_token)
        self.assertEqual(len(self.decoder.token_to_id), len(self.vocab_data))
    
    def test_get_token_ids(self):
        """Test token ID lookup."""
        token_id = self.decoder.get_token_ids("{")
        self.assertEqual(token_id, 5)
    
    def test_get_token_string(self):
        """Test token string lookup."""
        token_str = self.decoder.get_token_string(5)
        self.assertEqual(token_str, "{")
    
    def test_constrain_logits(self):
        """Test logit constraint."""
        logits = np.random.randn(len(self.vocab_data))
        constrained = self.decoder.constrain_logits(
            logits, "", self.functions
        )
        self.assertEqual(len(constrained), len(logits))
    
    def test_sample_next_token_greedy(self):
        """Test greedy sampling."""
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        token = self.decoder.sample_next_token(logits, temperature=0.0)
        self.assertEqual(token, 3)


if __name__ == '__main__':
    unittest.main()