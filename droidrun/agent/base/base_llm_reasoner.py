"""
Base LLM Reasoner - Base class for LLM-based reasoning.

This module provides the base class that all LLM reasoners must extend.
"""

import json
import re
import textwrap
import logging
from typing import Any, Dict, List, Optional, Union

from ..providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider
)

logger = logging.getLogger("droidrun")

class BaseLLMReasoner:
    """Base class for LLM-based reasoning with configurable prompts and response parsing."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        vision: bool = False,
        base_url: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None
    ):
        """Initialize the base LLM reasoner.
        
        Args:
            llm_provider: LLM provider name
            model_name: Model name to use
            api_key: API key for the LLM provider
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            vision: Whether vision capabilities are enabled
            base_url: Optional base URL for the API
            system_prompt_template: Optional template for system prompt
            user_prompt_template: Optional template for user prompt
            response_format: Optional expected format of the response
        """
        # Auto-detect Gemini models
        if model_name and model_name.startswith("gemini-"):
            llm_provider = "gemini"
            
        self.llm_provider = llm_provider.lower()
        
        # Initialize the appropriate provider
        provider_class = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "ollama": OllamaProvider
        }.get(self.llm_provider)
        
        if not provider_class:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        self.provider = provider_class(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            vision=vision,
            base_url=base_url
        )

        # Store templates
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.response_format = response_format or {}

    def get_token_usage_stats(self) -> Dict[str, int]:
        """Get current token usage statistics."""
        return self.provider.get_token_usage_stats()

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with the given arguments.
        
        Args:
            template: The prompt template string
            **kwargs: Arguments to format the template with
            
        Returns:
            Formatted prompt string
        """
        if not template:
            return ""
            
        try:
            return textwrap.dedent(template).format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template argument: {e}")
            return template
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return template

    def _create_system_prompt(self, **kwargs) -> str:
        """Create the system prompt using the template.
        
        Args:
            **kwargs: Arguments to format the template with
            
        Returns:
            Formatted system prompt
        """
        if not self.system_prompt_template:
            return ""
            
        return self._format_prompt(self.system_prompt_template, **kwargs)

    def _create_user_prompt(self, **kwargs) -> str:
        """Create the user prompt using the template.
        
        Args:
            **kwargs: Arguments to format the template with
            
        Returns:
            Formatted user prompt
        """
        if not self.user_prompt_template:
            return ""
            
        return self._format_prompt(self.user_prompt_template, **kwargs)

    def _parse_response(self, response: str) -> Union[Dict[str, Any], str]:
        """Parse the LLM response based on expected format.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Parsed response as dictionary or original string if no format specified
        """
        if not self.response_format:
            return response
            
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            
            # Ensure required fields are present
            for field, field_type in self.response_format.items():
                if field not in data:
                    if field_type == dict:
                        data[field] = {}
                    elif field_type == list:
                        data[field] = []
                    elif field_type == str:
                        data[field] = ""
                    elif field_type == int:
                        data[field] = 0
                    elif field_type == float:
                        data[field] = 0.0
                    elif field_type == bool:
                        data[field] = False
                        
            return data
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try regex for each field
            result = {}
            for field, _ in self.response_format.items():
                pattern = fr'{field}["\s:]+([^"]+)'
                match = re.search(pattern, response)
                result[field] = match.group(1) if match else None
                
            return result

    async def generate_response(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        screenshot_data: Optional[bytes] = None,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """Generate a response using the LLM.
        
        Args:
            system_prompt: Optional override for system prompt
            user_prompt: Optional override for user prompt
            screenshot_data: Optional screenshot data for vision tasks
            **kwargs: Additional arguments for prompt formatting
            
        Returns:
            Generated response (parsed if format specified)
        """
        # Use provided prompts or generate from templates
        final_system_prompt = system_prompt or self._create_system_prompt(**kwargs)
        final_user_prompt = user_prompt or self._create_user_prompt(**kwargs)
        
        try:
            # Get raw response from provider
            response = await self.provider.generate_response(
                system_prompt=final_system_prompt,
                user_prompt=final_user_prompt,
                screenshot_data=screenshot_data
            )
            
            # Parse response if format specified
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if self.response_format:
                return {field: None for field in self.response_format.keys()}
            return str(e)

    async def reason(
        self,
        goal: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[str]] = None,
        current_ui_state: Optional[str] = None,
        current_phone_state: Optional[str] = None,
        screenshot_data: Optional[bytes] = None,
        memories: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a reasoning step using the LLM.
        
        This is a convenience method that combines prompt creation and response generation
        for common reasoning scenarios.
        
        Args:
            goal: Optional goal or task description
            history: Optional list of previous steps
            available_tools: Optional list of available tools
            current_ui_state: Optional current UI state
            current_phone_state: Optional current phone state
            screenshot_data: Optional screenshot data
            memories: Optional list of memories
            **kwargs: Additional arguments for prompt formatting
            
        Returns:
            Reasoning result
        """
        # Combine all context into kwargs
        context = {
            "goal": goal,
            "history": history,
            "available_tools": available_tools,
            "current_ui_state": current_ui_state,
            "current_phone_state": current_phone_state,
            "memories": memories,
            **kwargs
        }
        
        # Generate response with all context
        return await self.generate_response(
            screenshot_data=screenshot_data,
            **context
        ) 