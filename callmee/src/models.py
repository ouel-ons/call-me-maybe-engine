"""Pydantic models for function definitions and test data."""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator


class FunctionParameter(BaseModel):
    """Model for a function parameter definition."""
    name: str
    type: Literal["number", "string", "boolean", "integer", "float"]
    description: Optional[str] = None
    required: bool = True


class FunctionDefinition(BaseModel):
    """Model for a function definition."""
    name: str
    description: str
    parameters: List[FunctionParameter]
    returns: Literal["number", "string", "boolean", "integer", "float", "void"]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionDefinition':
        """Create FunctionDefinition from various dict formats."""
        name = data.get('name', '')
        description = data.get('description', '')
        
        # Handle different parameter formats
        params = data.get('parameters', [])
        if isinstance(params, dict):
            # Format: {"param_name": {"type": "string"}}
            param_list = []
            for param_name, param_info in params.items():
                param_type = param_info.get('type', 'string')
                param_list.append(
                    FunctionParameter(
                        name=param_name,
                        type=param_type,
                        description=param_info.get('description', ''),
                        required=param_info.get('required', True)
                    )
                )
            params = param_list
        elif isinstance(params, list):
            # Format: [{"name": "a", "type": "number"}]
            param_list = []
            for param in params:
                if isinstance(param, dict):
                    if 'name' in param:
                        param_list.append(
                            FunctionParameter(
                                name=param.get('name', ''),
                                type=param.get('type', 'string'),
                                description=param.get('description', ''),
                                required=param.get('required', True)
                            )
                        )
                    else:
                        # Try to infer name from dict structure
                        for key, value in param.items():
                            if isinstance(value, dict) and 'type' in value:
                                param_list.append(
                                    FunctionParameter(
                                        name=key,
                                        type=value.get('type', 'string'),
                                        description=value.get('description', ''),
                                        required=value.get('required', True)
                                    )
                                )
            params = param_list
        
        # Handle returns format
        returns = data.get('returns', 'void')
        if isinstance(returns, dict):
            returns = returns.get('type', 'void')
        
        return cls(
            name=name,
            description=description,
            parameters=params if isinstance(params, list) else [],
            returns=returns
        )


class FunctionDefinitions(BaseModel):
    """Container for all function definitions."""
    functions: List[FunctionDefinition]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> 'FunctionDefinitions':
        """Create FunctionDefinitions from a list of function dicts."""
        functions = []
        for func_data in data:
            functions.append(FunctionDefinition.from_dict(func_data))
        return cls(functions=functions)


class FunctionCallResult(BaseModel):
    """Model for a single function call result."""
    prompt: str
    name: str
    parameters: Dict[str, Any]

    @validator('parameters')
    def validate_parameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameters is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError('parameters must be a dictionary')
        return v


class FunctionCallResults(BaseModel):
    """Container for all function call results."""
    results: List[FunctionCallResult]