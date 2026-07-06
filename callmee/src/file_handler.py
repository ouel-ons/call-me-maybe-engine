"""File I/O handling for the function calling system."""

import json
import os
from typing import List, Dict, Any, Optional

from .models import FunctionDefinition, FunctionDefinitions, FunctionCallResult


class FileHandler:
    """Handles file operations for the function calling system."""
    
    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        """
        Load and parse a JSON file with error handling.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            Parsed JSON data.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file contains invalid JSON.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {file_path}: {e.msg}",
                e.doc,
                e.pos
            )
    
    @staticmethod
    def load_function_definitions(file_path: str) -> List[FunctionDefinition]:
        """
        Load function definitions from a JSON file.
        
        Args:
            file_path: Path to the function definitions file.
            
        Returns:
            List of FunctionDefinition objects.
            
        Raises:
            ValueError: If the file format is invalid.
        """
        data = FileHandler.load_json_file(file_path)
        
        # Check if data is a list (direct array of functions)
        if isinstance(data, list):
            definitions = FunctionDefinitions.from_list(data)
            return definitions.functions
        
        # Check if data has 'functions' key
        if isinstance(data, dict) and 'functions' in data:
            if isinstance(data['functions'], list):
                definitions = FunctionDefinitions.from_list(data['functions'])
                return definitions.functions
        
        raise ValueError("Invalid function definitions format. Expected a list of functions or object with 'functions' key.")
    
    @staticmethod
    def load_prompts(file_path: str) -> List[str]:
        """
        Load prompts from a JSON file.
        
        Args:
            file_path: Path to the prompts file.
            
        Returns:
            List of prompt strings.
            
        Raises:
            ValueError: If the file format is invalid.
        """
        data = FileHandler.load_json_file(file_path)
        
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of prompts")
        
        prompts = []
        for item in data:
            if isinstance(item, dict) and 'prompt' in item:
                prompts.append(item['prompt'])
            elif isinstance(item, str):
                prompts.append(item)
            else:
                prompts.append(str(item))
        
        return prompts
    
    @staticmethod
    def save_results(
        results: List[FunctionCallResult],
        output_path: str
    ) -> None:
        """
        Save function call results to a JSON file.
        
        Args:
            results: List of FunctionCallResult objects.
            output_path: Path to the output file.
            
        Raises:
            IOError: If writing fails.
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert to dictionary
        results_data = []
        for result in results:
            results_data.append({
                "prompt": result.prompt,
                "name": result.name,
                "parameters": result.parameters
            })
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Failed to write output file: {e}")
    
    @staticmethod
    def validate_output(results: List[FunctionCallResult]) -> bool:
        """
        Validate the results against the expected format.
        
        Args:
            results: List of FunctionCallResult objects.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            for result in results:
                # Check required fields
                if not hasattr(result, 'prompt') or not result.prompt:
                    return False
                if not hasattr(result, 'name') or not result.name:
                    return False
                if not hasattr(result, 'parameters'):
                    return False
                if not isinstance(result.parameters, dict):
                    return False
            return True
        except:
            return False