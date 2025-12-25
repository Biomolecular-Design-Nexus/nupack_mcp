"""
Shared library for NUPACK MCP scripts.

This library contains common functions extracted from the individual scripts
to reduce code duplication while maintaining independence.
"""

__version__ = "1.0.0"

from .io import load_json, save_json, load_sequences, save_sequences
from .structure_utils import extract_base_pairs, validate_structure, get_gc_content
from .config_utils import load_config, merge_configs