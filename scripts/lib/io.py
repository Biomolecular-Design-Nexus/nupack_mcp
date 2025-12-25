"""
Shared I/O functions for MCP scripts.

These are extracted and simplified from common patterns across all scripts.
"""

from pathlib import Path
from typing import Union, Any, List
import json


def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file with error handling."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """Save data to JSON file with directory creation."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_sequences(file_path: Union[str, Path]) -> List[str]:
    """Load sequences from file (one per line)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {file_path}")

    with open(file_path) as f:
        sequences = [line.strip() for line in f if line.strip()]

    if not sequences:
        raise ValueError(f"No sequences found in {file_path}")

    return sequences


def save_sequences(sequences: List[str], file_path: Union[str, Path]) -> None:
    """Save sequences to file (one per line)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")


def ensure_output_dir(file_path: Union[str, Path]) -> Path:
    """Ensure output directory exists and return Path object."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path