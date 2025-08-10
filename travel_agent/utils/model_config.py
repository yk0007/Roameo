"""Utility module for handling model configuration and initialization."""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelConfig:
    """Handles model configuration and initialization based on environment variables."""
    
    # Supported providers and their environment variable names
    PROVIDERS = {
        'gemini': {
            'api_key_var': 'GOOGLE_API_KEY',
            'default_model_var': 'GEMINI_DEFAULT_MODEL',
            'models': {
                'gemini-1.5-flash': 'gemini-1.5-flash',
                'gemini-1.5-pro': 'gemini-1.5-pro',
                # Add more Gemini models as needed
            }
        },
        'groq': {
            'api_key_var': 'GROQ_API_KEY',
            'default_model_var': 'GROQ_DEFAULT_MODEL',
            'models': {
                'mixtral-8x7b-32768': 'mixtral-8x7b-32768',
                'llama2-70b-4096': 'llama2-70b-4096',
                'gemma-7b-it': 'gemma-7b-it',
                # Add more Groq models as needed
            }
        }
    }
    
    @classmethod
    def get_provider(cls) -> str:
        """Get the current model provider from environment variables."""
        provider = os.getenv('MODEL_PROVIDER', 'groq').lower()
        if provider not in cls.PROVIDERS:
            print(f"Warning: Unknown provider '{provider}'. Defaulting to 'groq'.")
            return 'groq'
        return provider
    
    @classmethod
    def get_model_name(cls, provider: Optional[str] = None) -> str:
        """Get the model name for the specified provider."""
        if provider is None:
            provider = cls.get_provider()
            
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
            
        provider_config = cls.PROVIDERS[provider]
        model_name = os.getenv(provider_config['default_model_var'])
        
        if not model_name:
            # Default to first available model if not specified
            model_name = next(iter(provider_config['models'].values()))
            
        return model_name
    
    @classmethod
    def get_api_key(cls, provider: Optional[str] = None) -> str:
        """Get the API key for the specified provider."""
        if provider is None:
            provider = cls.get_provider()
            
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
            
        api_key = os.getenv(cls.PROVIDERS[provider]['api_key_var'])
        if not api_key:
            raise ValueError(
                f"API key for {provider} not found. "
                f"Please set the {cls.PROVIDERS[provider]['api_key_var']} environment variable."
            )
            
        return api_key
    
    @classmethod
    def get_model_config(cls, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get the complete configuration for the specified provider."""
        if provider is None:
            provider = cls.get_provider()
            
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
            
        return {
            'provider': provider,
            'model_name': cls.get_model_name(provider),
            'api_key': cls.get_api_key(provider),
            'models': cls.PROVIDERS[provider]['models']
        }
    
    @classmethod
    def get_llm_instance(cls, **kwargs):
        """Get an instance of the LLM based on the current configuration."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq
        
        # Get the current provider and configuration
        provider = cls.get_provider()
        model_name = cls.get_model_name(provider)
        api_key = cls.get_api_key(provider)
        
        print(f"DEBUG: Initializing {provider} model: {model_name}")  # Debug log
        
        # Set default temperature if not provided
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.3
            
        try:
            if provider == 'gemini':
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    **kwargs
                )
            elif provider == 'groq':
                return ChatGroq(
                    model_name=model_name,
                    groq_api_key=api_key,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            print(f"ERROR: Failed to initialize {provider} model: {str(e)}")
            raise
