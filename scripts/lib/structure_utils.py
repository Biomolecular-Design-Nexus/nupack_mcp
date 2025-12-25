"""
Shared structure analysis utilities.

Common functions for working with secondary structures and sequences.
"""

from typing import List, Tuple


def extract_base_pairs(structure: str) -> List[Tuple[int, int]]:
    """
    Extract base pairs from dot-bracket notation.

    Args:
        structure: Secondary structure in dot-bracket notation

    Returns:
        List of (i, j) base pair tuples
    """
    pairs = []
    stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))

    return pairs


def validate_structure(structure: str) -> bool:
    """
    Validate that structure has balanced brackets.

    Args:
        structure: Secondary structure in dot-bracket notation

    Returns:
        True if structure is valid
    """
    if not structure:
        return True

    # Check only allowed characters
    if not all(c in '().' for c in structure):
        return False

    # Check bracket balance
    depth = 0
    for char in structure:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return False

    return depth == 0


def get_gc_content(sequence: str) -> float:
    """
    Calculate GC content of nucleic acid sequence.

    Args:
        sequence: DNA or RNA sequence

    Returns:
        GC content as fraction (0.0 to 1.0)
    """
    if not sequence:
        return 0.0

    gc_count = sum(1 for base in sequence.upper() if base in ['G', 'C'])
    return gc_count / len(sequence)


def complement_base(base: str, material: str = "RNA") -> str:
    """
    Get complement base for given nucleotide.

    Args:
        base: Nucleotide base (A, T, G, C, U)
        material: "DNA" or "RNA"

    Returns:
        Complement base
    """
    if material.upper() == "DNA":
        complements = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    else:  # RNA
        complements = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

    return complements.get(base.upper(), 'N')


def can_pair(base1: str, base2: str, material: str = "RNA") -> bool:
    """
    Check if two bases can form Watson-Crick pairs.

    Args:
        base1: First base
        base2: Second base
        material: "DNA" or "RNA"

    Returns:
        True if bases can pair
    """
    if material.upper() == "DNA":
        valid_pairs = [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')]
    else:  # RNA
        valid_pairs = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]

    return (base1.upper(), base2.upper()) in valid_pairs