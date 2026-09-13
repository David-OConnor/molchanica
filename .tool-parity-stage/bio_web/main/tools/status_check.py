"""Compatibility import for the shared bio_tools implementation."""
import sys
from bio_tool_adapters import status_check as _implementation
sys.modules[__name__] = _implementation
