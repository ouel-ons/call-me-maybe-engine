"""Main entry point for the function calling system."""

import sys
import os
import argparse
import json
from typing import Dict, Any

# Add the project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Now import llm_sdk
try:
    from llm_sdk import Small_LLM_Model
    print("✅ llm_sdk imported successfully")
except ImportError as e:
    print(f"❌ Error importing llm_sdk: {e}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Files in project root: {os.listdir(PROJECT_ROOT)}")
    sys.exit(1)

from .config import Config
from .file_handler import FileHandler
from .function_caller import FunctionCaller


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Function calling system with constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to function definitions file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to output results file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    return vars(args)


def main() -> int:
    """
    Main entry point for the function calling system.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create configuration
        config = Config.from_args(args)
        config.ensure_directories()
        
        print(f"Loading function definitions from: {config.functions_definition_file}")
        functions = FileHandler.load_function_definitions(
            config.functions_definition_file
        )
        print(f"Loaded {len(functions)} function definitions")
        
        print(f"Loading prompts from: {config.input_file}")
        prompts = FileHandler.load_prompts(config.input_file)
        print(f"Loaded {len(prompts)} prompts")
        
        print("Initializing LLM model...")
        model = Small_LLM_Model()
        print("LLM model initialized")
        
        print("Initializing function caller...")
        caller = FunctionCaller(model, functions)
        print("Function caller initialized")
        
        print(f"Processing {len(prompts)} prompts...")
        results = caller.process_all(prompts)
        print(f"Processed {len(results)} prompts")
        
        print("Validating results...")
        if not FileHandler.validate_output(results):
            print("Warning: Output validation failed")
        
        print(f"Saving results to: {config.output_file}")
        FileHandler.save_results(results, config.output_file)
        print("Done!")
        
        print("\n=== Summary ===")
        for i, result in enumerate(results):
            params_str = ", ".join(f"{k}: {v}" for k, v in result.parameters.items())
            print(f"{i+1}. {result.prompt[:50]}... -> {result.name}({params_str})")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.get('debug', False):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
