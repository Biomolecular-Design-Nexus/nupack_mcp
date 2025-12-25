#!/usr/bin/env python3
"""
Script: sequence_design.py
Description: Design nucleic acid sequences to fold into target structures

Original Use Case: examples/use_case_3_sequence_design.py
Dependencies Removed: None (already self-contained)

Usage:
    python scripts/sequence_design.py --input <structure_file> --output <output_file>
    python scripts/sequence_design.py --target-structure <structure> --output <output_file>

Example:
    python scripts/sequence_design.py --target-structure "((((....))))" --num-designs 5 --output results/design.json
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
import random

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    "material": "DNA",
    "temperature": 37.0,
    "num_designs": 5,
    "max_attempts": 1000,
    "defects_threshold": 2,
    "gc_content_range": [0.3, 0.7]
}

# ==============================================================================
# Core Classes
# ==============================================================================
class TargetStructure:
    """Target structure for sequence design"""
    def __init__(self, structure: str):
        self.structure = structure
        self.length = len(structure)
        self.base_pairs = self._extract_base_pairs()
        self.unpaired_positions = self._get_unpaired_positions()

    def _extract_base_pairs(self) -> List[Tuple[int, int]]:
        """Extract base pairs from dot-bracket notation"""
        pairs = []
        stack = []

        for i, char in enumerate(self.structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))

        return pairs

    def _get_unpaired_positions(self) -> List[int]:
        """Get positions that are unpaired (dots)"""
        return [i for i, char in enumerate(self.structure) if char == '.']

class DesignCandidate:
    """Container for a designed sequence candidate"""
    def __init__(self, sequence: str, target_structure: str, score: float = 0.0):
        self.sequence = sequence
        self.target_structure = target_structure
        self.score = score
        self.defects = 0
        self.gc_content = self._calculate_gc_content()

    def _calculate_gc_content(self) -> float:
        """Calculate GC content of sequence"""
        gc_count = sum(1 for base in self.sequence if base in ['G', 'C'])
        return gc_count / len(self.sequence)

# ==============================================================================
# Core Functions
# ==============================================================================
def generate_base_paired_sequence(target: TargetStructure, material: str = "DNA") -> str:
    """
    Generate a sequence that satisfies base pairing constraints.
    Simplified from NUPACK design algorithms.
    """
    # Base pairing rules
    if material == "DNA":
        pairs = [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')]
        bases = ['A', 'T', 'G', 'C']
    else:  # RNA
        pairs = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]
        bases = ['A', 'U', 'G', 'C']

    complement = dict(pairs)
    sequence = ['N'] * target.length

    # Assign complementary bases to paired positions
    for i, j in target.base_pairs:
        if sequence[i] == 'N' and sequence[j] == 'N':
            # Choose a random base pair
            pair = random.choice(pairs)
            sequence[i] = pair[0]
            sequence[j] = pair[1]
        elif sequence[i] == 'N':
            sequence[i] = complement.get(sequence[j], random.choice(bases))
        elif sequence[j] == 'N':
            sequence[j] = complement.get(sequence[i], random.choice(bases))

    # Fill unpaired positions with random bases
    for i in target.unpaired_positions:
        if sequence[i] == 'N':
            sequence[i] = random.choice(bases)

    return ''.join(sequence)

def calculate_design_score(candidate: DesignCandidate, target: TargetStructure) -> float:
    """
    Calculate design score based on sequence quality.
    Lower scores are better (representing defects or energy).
    """
    score = 0.0
    defects = 0

    # Check base pairing constraints
    for i, j in target.base_pairs:
        base_i = candidate.sequence[i]
        base_j = candidate.sequence[j]

        # Check if bases can form Watson-Crick pairs
        valid_pairs = [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G'),
                      ('A', 'U'), ('U', 'A')]

        if (base_i, base_j) not in valid_pairs:
            defects += 1
            score += 1.0  # Penalty for non-Watson-Crick pairs

    # GC content penalty (prefer balanced GC content)
    gc_optimal = 0.5
    gc_penalty = abs(candidate.gc_content - gc_optimal) * 2
    score += gc_penalty

    # Sequence complexity penalty (avoid repeats)
    complexity_penalty = 0
    for i in range(len(candidate.sequence) - 2):
        triplet = candidate.sequence[i:i+3]
        if triplet == triplet[0] * 3:  # Homopolymer
            complexity_penalty += 0.5

    score += complexity_penalty

    candidate.defects = defects
    candidate.score = score
    return score

def design_sequences(target_structure: str, num_designs: int = 5, material: str = "DNA",
                    max_attempts: int = 1000) -> List[DesignCandidate]:
    """
    Design multiple sequences for target structure.
    Simplified from NUPACK design() functionality.
    """
    target = TargetStructure(target_structure)
    designs = []

    attempts = 0
    while len(designs) < num_designs and attempts < max_attempts:
        attempts += 1

        # Generate candidate sequence
        sequence = generate_base_paired_sequence(target, material)
        candidate = DesignCandidate(sequence, target_structure)

        # Calculate score
        calculate_design_score(candidate, target)

        # Accept candidate if it's reasonable
        if candidate.defects <= 2:  # Allow some flexibility
            designs.append(candidate)

    # Sort by score (lower is better)
    designs.sort(key=lambda x: x.score)

    return designs

def analyze_design_quality(designs: List[DesignCandidate]) -> Dict[str, Any]:
    """Analyze overall quality of designs"""
    if not designs:
        return {"error": "No designs generated"}

    scores = [d.score for d in designs]
    defects = [d.defects for d in designs]
    gc_contents = [d.gc_content for d in designs]

    return {
        "total_designs": len(designs),
        "score_statistics": {
            "mean": np.mean(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "std": np.std(scores)
        },
        "defects_statistics": {
            "mean": np.mean(defects),
            "min": int(np.min(defects)),
            "max": int(np.max(defects))
        },
        "gc_content_statistics": {
            "mean": np.mean(gc_contents),
            "min": np.min(gc_contents),
            "max": np.max(gc_contents),
            "std": np.std(gc_contents)
        }
    }

# ==============================================================================
# Main Function (MCP-ready)
# ==============================================================================
def run_sequence_design(
    target_structure: Optional[str] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for sequence design.

    Args:
        target_structure: Target secondary structure in dot-bracket notation
        input_file: Path to file with target structure
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - designs: List of designed sequences with scores
            - target_analysis: Analysis of target structure
            - quality_analysis: Overall design quality statistics
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_sequence_design(
        ...     target_structure="((((....))))",
        ...     num_designs=5,
        ...     output_file="design.json"
        ... )
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Get target structure from input
    if input_file:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        with open(input_file) as f:
            target_structure = f.read().strip()

    if not target_structure:
        raise ValueError("No target structure provided")

    # Validate structure notation
    if not all(c in '().{}[]' for c in target_structure):
        raise ValueError("Invalid structure notation. Use dot-bracket notation: ().")

    # Create target structure object
    target = TargetStructure(target_structure)

    # Design sequences
    designs = design_sequences(
        target_structure,
        num_designs=config["num_designs"],
        material=config["material"],
        max_attempts=config["max_attempts"]
    )

    # Analyze design quality
    quality_analysis = analyze_design_quality(designs)

    # Prepare design results
    design_results = []
    for i, design in enumerate(designs):
        design_results.append({
            "rank": i + 1,
            "sequence": design.sequence,
            "score": round(design.score, 3),
            "defects": design.defects,
            "gc_content": round(design.gc_content, 3),
            "length": len(design.sequence)
        })

    # Target structure analysis
    target_analysis = {
        "structure": target.structure,
        "length": target.length,
        "base_pairs": target.base_pairs,
        "num_pairs": len(target.base_pairs),
        "unpaired_positions": len(target.unpaired_positions)
    }

    # Prepare output data
    output_data = {
        "analysis_type": "sequence_design",
        "model_parameters": {
            "material": config["material"],
            "temperature": config["temperature"],
            "num_designs": config["num_designs"]
        },
        "target_structure": target_analysis,
        "designs": design_results,
        "quality_analysis": quality_analysis
    }

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    return {
        "designs": design_results,
        "target_analysis": target_analysis,
        "quality_analysis": quality_analysis,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "target_length": target.length,
            "designs_generated": len(designs),
            "config": config
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--target-structure',
                       help='Target structure in dot-bracket notation')
    parser.add_argument('--input', '-i',
                       help='Input file with target structure')
    parser.add_argument('--output', '-o',
                       help='Output file path')
    parser.add_argument('--material', choices=['DNA', 'RNA'], default='DNA',
                       help='Material type (DNA or RNA)')
    parser.add_argument('--temperature', type=float, default=37.0,
                       help='Temperature in Celsius')
    parser.add_argument('--num-designs', type=int, default=5,
                       help='Number of designs to generate')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    kwargs = {}
    if args.material != 'DNA':
        kwargs['material'] = args.material
    if args.temperature != 37.0:
        kwargs['temperature'] = args.temperature
    if args.num_designs != 5:
        kwargs['num_designs'] = args.num_designs

    # Run design
    result = run_sequence_design(
        target_structure=args.target_structure,
        input_file=args.input,
        output_file=args.output,
        config=config,
        **kwargs
    )

    print(f"✅ Sequence design completed")
    print(f"   Target length: {result['metadata']['target_length']}")
    print(f"   Designs generated: {result['metadata']['designs_generated']}")
    if result['output_file']:
        print(f"   Results saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()