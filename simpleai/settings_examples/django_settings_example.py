"""Example Django settings for SimpleAI."""

SIMPLEAI = {
    "defaults": ["gemini", "openai", "claude", "grok", "perplexity"],
    "providers": {
        "gemini": {
            "api_key": "YOUR_GEMINI_API_KEY",
            "default_model": "gemini-3-pro-preview",
            "max_output_tokens": 8192,
        },
        "claude": {
            "api_key": "YOUR_ANTHROPIC_API_KEY",
            "default_model": "claude-opus-4-6",
            "max_tokens": 4096,
            "max_retries": 3,  # retries on 429 errors (uses retry-after header)
            "skip_citation_followup": False,  # skip extra API call for citations
        },
        "openai": {
            "api_key": "YOUR_OPENAI_API_KEY",
            "default_model": "gpt-5.2",
            "max_output_tokens": 8192,
            "base_url": None,
        },
        "grok": {
            "api_key": "YOUR_XAI_API_KEY",
            "default_model": "grok-4-latest",
            "max_tokens": 8192,
        },
        "perplexity": {
            "api_key": "YOUR_PERPLEXITY_API_KEY",
            "default_model": "sonar-deep-research",
            "max_output_tokens": 4096,
        },
    },
    "logging": {
        "enabled": True,
        "network_logging": True,
        "django_logfile": "django",
        "logfile_location": "./simpleai.log",
    },
}
