"""Configuration management for the function calling system."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for the function calling system."""
    
    # File paths
    functions_definition_file: str = "data/input/functions_definition.json"
    input_file: str = "data/input/function_calling_tests.json"
    output_file: str = "data/output/function_calling_results.json"
    
    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.0  # Deterministic output
    
    # Model settings
    model_name: str = "Qwen/Qwen3-0.6B"
    
    @classmethod
    def from_args(cls, args: Optional[dict] = None) -> 'Config':
        """Create config from command line arguments."""
        config = cls()
        if args:
            if args.get('functions_definition'):
                config.functions_definition_file = args['functions_definition']
            if args.get('input'):
                config.input_file = args['input']
            if args.get('output'):
                config.output_file = args['output']
        return config
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.input_file, self.output_file]:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)