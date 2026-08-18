"""Function calling system with LLM integration and constrained decoding."""

import json
import re
from typing import List, Dict, Any, Optional
import numpy as np

from .models import FunctionDefinition, FunctionCallResult
from .decoder import ConstrainedDecoder


class FunctionCaller:
    """
    Function calling system that uses constrained decoding to generate
    valid function calls from natural language prompts.
    """
    
    def __init__(self, llm_model, functions: List[FunctionDefinition]):
        """
        Initialize the function caller.
        
        Args:
            llm_model: The LLM model instance.
            functions: List of available function definitions.
        """
        self.model = llm_model
        self.functions = functions
        
        # Initialize decoder with vocabulary
        try:
            vocab_path = self.model.get_path_to_vocab_file()
            self.decoder = ConstrainedDecoder(vocab_path)
        except Exception as e:
            print(f"Warning: Could not initialize decoder: {e}")
            # Create a fallback decoder with a dummy vocab
            import tempfile
            import json
            temp_vocab = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vocab = {str(i): i for i in range(1000)}
            json.dump(vocab, temp_vocab)
            temp_vocab.close()
            self.decoder = ConstrainedDecoder(temp_vocab.name)
        
    def call_function(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> FunctionCallResult:
        """
        Process a natural language prompt and generate a function call.
        
        Args:
            prompt: Natural language request.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            FunctionCallResult with function name and parameters.
        """
        # Build the prompt for the LLM
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser request: {prompt}\n\nFunction call:"
        
        # Tokenize the prompt
        try:
            input_ids = self.model.encode(full_prompt)
        except Exception as e:
            print(f"Error encoding prompt: {e}")
            # Fallback: try to extract function call from prompt
            return self._fallback_parse(prompt, prompt)
        
        # Handle different return types from encode
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        elif isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()
        
        # If it's a nested list, flatten
        if isinstance(input_ids, list) and len(input_ids) > 0:
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]
        
        # Ensure input_ids is a list of ints
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)
        
        # Generate with constrained decoding
        generated_text = self._generate_with_constraints(
            input_ids, max_tokens, temperature
        )
        
        # Parse the generated function call
        return self._parse_function_call(prompt, generated_text)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with function definitions."""
        prompt = "You are a function calling system. Given a user request, "
        prompt += "output a JSON object with 'function' and 'arguments' keys.\n\n"
        prompt += "Available functions:\n"
        
        for func in self.functions:
            params_str = ', '.join([f"{p.name}: {p.type}" for p in func.parameters])
            prompt += f"- {func.name}({params_str}): {func.description}\n"
        
        prompt += "\nOutput format: {\"function\": \"function_name\", \"arguments\": {\"param1\": value, ...}}\n"
        prompt += "Only output the JSON object, nothing else."
        return prompt
    
    def _generate_with_constraints(
        self,
        input_ids: List[int],
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Generate text with constrained decoding.
        
        Args:
            input_ids: List of input token IDs.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text string.
        """
        generated_tokens = []
        current_output = ""
        
        for _ in range(max_tokens):
            try:
                # Get logits from the model
                logits = self.model.get_logits_from_input_ids(input_ids)
            except Exception as e:
                print(f"Error getting logits: {e}")
                break
            
            # Convert to numpy array if needed
            if isinstance(logits, list):
                logits = np.array(logits)
            elif hasattr(logits, 'numpy'):
                logits = logits.numpy()
            
            # Ensure logits is a 1D array
            if logits.ndim > 1:
                logits = logits.flatten()
            
            # Apply constraints
            constrained_logits = self.decoder.constrain_logits(
                logits, current_output, self.functions
            )
            
            # Sample next token
            next_token = self.decoder.sample_next_token(
                constrained_logits, temperature
            )
            
            # Add token to input for next step
            input_ids.append(next_token)
            
            # Get token string
            token_str = self.decoder.get_token_string(next_token)
            if token_str:
                generated_tokens.append(token_str)
                current_output += token_str
                
                # Check for end of generation (complete JSON object)
                if self._is_complete_json(current_output):
                    break
                
        return current_output
    
    def _is_complete_json(self, text: str) -> bool:
        """Check if text contains complete JSON."""
        try:
            # Find the first opening brace
            start = text.find('{')
            if start == -1:
                return False
            
            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        try:
                            json.loads(text[start:i+1])
                            return True
                        except:
                            return False
            return False
        except:
            return False
    
    def _parse_function_call(
        self,
        prompt: str,
        generated_text: str
    ) -> FunctionCallResult:
        """
        Parse the generated text into a function call result.
        
        Args:
            prompt: Original prompt.
            generated_text: Generated text from the model.
            
        Returns:
            FunctionCallResult object.
        """
        try:
            # Find JSON in the generated text
            start = generated_text.find('{')
            if start == -1:
                raise ValueError("No JSON found in generated text")
            
            # Extract complete JSON object
            json_str = self._extract_json(generated_text[start:])
            data = json.loads(json_str)
            
            # Extract function and arguments
            function_name = data.get('function', '')
            arguments = data.get('arguments', {})
            
            # Validate function name
            func_def = self._find_function_definition(function_name)
            if func_def is None:
                # Try to find a matching function
                function_name = self._find_matching_function(function_name)
                if function_name is None:
                    func_def = self.functions[0]
                    function_name = func_def.name
                else:
                    func_def = self._find_function_definition(function_name)
            
            # Ensure all required arguments are present
            if func_def:
                arguments = self._ensure_required_args(prompt, arguments, func_def)
            
            return FunctionCallResult(
                prompt=prompt,
                name=function_name,
                parameters=arguments
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract function call from text
            return self._fallback_parse(prompt, generated_text)
    
    def _extract_json(self, text: str) -> str:
        """Extract a complete JSON object from text."""
        brace_count = 0
        for i, char in enumerate(text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[:i+1]
        return text
    
    def _find_function_definition(self, name: str) -> Optional[FunctionDefinition]:
        """Find function definition by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None
    
    def _find_matching_function(self, name: str) -> Optional[str]:
        """Find the closest matching function name."""
        # Try exact match first
        for func in self.functions:
            if func.name == name:
                return func.name
        
        # Try case-insensitive match
        name_lower = name.lower()
        for func in self.functions:
            if func.name.lower() == name_lower:
                return func.name
        
        # Try partial match
        for func in self.functions:
            if name_lower in func.name.lower() or func.name.lower() in name_lower:
                return func.name
        
        return None
    
    def _ensure_required_args(
        self,
        prompt: str,
        arguments: Dict[str, Any],
        func_def: FunctionDefinition
    ) -> Dict[str, Any]:
        """Ensure all required arguments are present."""
        for param in func_def.parameters:
            if param.required and param.name not in arguments:
                # Try to infer the argument from the prompt
                value = self._infer_argument(prompt, param)
                if value is not None:
                    arguments[param.name] = value
        return arguments
    
    def _infer_argument(self, prompt: str, param) -> Optional[Any]:
        """Try to infer a missing argument from the prompt."""
        if param.type in ['number', 'integer', 'float']:
            # Look for numbers in the prompt
            numbers = re.findall(r'-?\d+\.?\d*', prompt)
            if numbers:
                try:
                    if param.type == 'integer':
                        return int(float(numbers[0]))
                    else:
                        return float(numbers[0])
                except:
                    pass
        return None
    
    def _fallback_parse(self, prompt: str, text: str) -> FunctionCallResult:
        """Fallback parsing when JSON extraction fails."""
        # Try to find function name
        function_name = self.functions[0].name
        for func in self.functions:
            if func.name.lower() in text.lower():
                function_name = func.name
                break
        
        # Try to find arguments
        arguments = {}
        func_def = self._find_function_definition(function_name)
        if func_def:
            for param in func_def.parameters:
                value = self._infer_argument(prompt, param)
                if value is not None:
                    arguments[param.name] = value
        
        return FunctionCallResult(
            prompt=prompt,
            name=function_name,
            parameters=arguments
        )
    
    def process_all(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> List[FunctionCallResult]:
        """
        Process multiple prompts.
        
        Args:
            prompts: List of natural language prompts.
            max_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
            
        Returns:
            List of FunctionCallResult objects.
        """
        results = []
        for prompt in prompts:
            try:
                result = self.call_function(prompt, max_tokens, temperature)
                results.append(result)
            except Exception as e:
                # Graceful fallback for errors
                results.append(
                    FunctionCallResult(
                        prompt=prompt,
                        name="unknown",
                        parameters={}
                    )
                )
        return results