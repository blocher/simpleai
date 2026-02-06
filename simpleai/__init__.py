"""SimpleAI public API."""

from __future__ import annotations

from .api import run_prompt
from .exceptions import SimpleAIException

__all__ = ["run_prompt", "SimpleAIException"]
__version__ = "0.1.0"
