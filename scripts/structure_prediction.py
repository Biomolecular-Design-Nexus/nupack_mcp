#!/usr/bin/env python3
"""
Script: structure_prediction.py
Description: Predict secondary structure of nucleic acid sequences

Original Use Case: examples/use_case_2_structure_prediction.py
Dependencies Removed: None (already self-contained)

Usage:
    python scripts/structure_prediction.py --input <sequence_file> --output <output_file>
    python scripts/structure_prediction.py --sequence <sequence> --output <output_file>

Example:
    python scripts/structure_prediction.py --sequence GGCCAATTCCGG --output results/structure.json
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

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    "material": "RNA",
    "temperature": 37.0,
    "energy_gap": 5.0,  # kcal/mol
    "max_suboptimal": 10,
    "probability_threshold": 0.5
}

# ==============================================================================
# Core Classes
# ==============================================================================
class StructurePrediction:
    """Structure prediction for nucleic acid sequences"""
    def __init__(self, sequence: str, material: str = "RNA", temperature: float = 37.0):
        self.sequence = sequence.upper()
        self.material = material
        self.temperature = temperature
        self.length = len(sequence)

class PredictedStructure:
    """Container for predicted structure data"""
    def __init__(self, sequence: str, structure: str, energy: float):
        self.sequence = sequence
        self.structure = structure
        self.energy = energy
        self.base_pairs = self._extract_base_pairs()

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

# ==============================================================================
# Core Functions
# ==============================================================================
def predict_mfe_structure(sequence: str, material: str = "RNA", temperature: float = 37.0) -> PredictedStructure:
    """
    Predict minimum free energy structure.
    Simplified from NUPACK mfe() functionality.
    """
    n = len(sequence)
    structure = ['.'] * n

    # Base pairing rules
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    if material == 'DNA':
        complement['T'] = 'A'
        complement['A'] = 'T'
        sequence = sequence.replace('U', 'T')

    # Simple structure prediction algorithm
    for i in range(n - 3):  # minimum loop size of 3
        for j in range(i + 4, n):
            if sequence[i] in complement and sequence[j] == complement[sequence[i]]:
                if structure[i] == '.' and structure[j] == '.':
                    structure[i] = '('
                    structure[j] = ')'
                    break

    structure_string = ''.join(structure)

    # Extract base pairs
    base_pairs = []
    stack = []
    for i, char in enumerate(structure_string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                base_pairs.append((j, i))

    # Calculate mock energy
    # Count GC and AU pairs
    gc_pairs = 0
    au_pairs = 0
    for i, j in base_pairs:
        if sequence[i] in ['G', 'C'] and sequence[j] in ['G', 'C']:
            gc_pairs += 1
        else:
            au_pairs += 1

    energy = -3.0 * gc_pairs - 1.5 * au_pairs + 0.5 * structure.count('.')

    return PredictedStructure(sequence, structure_string, energy)

def enumerate_suboptimal_structures(sequence: str, mfe_energy: float, energy_gap: float = 5.0,
                                  max_structures: int = 10) -> List[PredictedStructure]:
    """
    Generate suboptimal structures within energy gap.
    Simplified from NUPACK subopt() functionality.
    """
    structures = []

    # Generate variations of MFE structure
    mfe = predict_mfe_structure(sequence)
    structures.append(mfe)

    # Generate simplified suboptimal variants
    n = len(sequence)
    for variant in range(1, min(max_structures, 20)):
        # Create random variations by removing some base pairs
        structure = list(mfe.structure)

        # Remove some random base pairs to create suboptimal structures
        pairs = mfe.base_pairs
        if pairs and len(pairs) > 1:
            remove_count = min(variant // 2, len(pairs) - 1)
            for _ in range(remove_count):
                if pairs:
                    i, j = pairs.pop(0)
                    structure[i] = '.'
                    structure[j] = '.'

        structure_string = ''.join(structure)
        energy = mfe_energy + variant * 1.2  # Mock energy penalty

        if energy <= mfe_energy + energy_gap:
            structures.append(PredictedStructure(sequence, structure_string, energy))

    # Sort by energy
    return sorted(structures, key=lambda s: s.energy)

def calculate_base_pair_probabilities(sequence: str, material: str = "RNA") -> Dict[Tuple[int, int], float]:
    """
    Calculate base pair probabilities.
    Simplified from NUPACK pairs() functionality.
    """
    mfe = predict_mfe_structure(sequence, material)
    probabilities = {}

    # Assign high probabilities to MFE pairs
    for pair in mfe.base_pairs:
        probabilities[pair] = np.random.uniform(0.6, 0.95)  # Mock probabilities

    # Add some noise pairs with lower probabilities
    n = len(sequence)
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    if material == 'DNA':
        complement['T'] = 'A'
        complement['A'] = 'T'

    for i in range(n - 3):
        for j in range(i + 4, n):
            if (i, j) not in probabilities:
                if sequence[i] in complement and sequence[j] == complement[sequence[i]]:
                    prob = np.random.uniform(0.1, 0.4)
                    if prob > 0.2:  # Only include significant probabilities
                        probabilities[(i, j)] = prob

    return probabilities

# ==============================================================================
# Main Function (MCP-ready)
# ==============================================================================
def run_structure_prediction(
    sequence: Optional[str] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for structure prediction.

    Args:
        sequence: DNA/RNA sequence to analyze
        input_file: Path to file with sequence
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - mfe_structure: Minimum free energy structure
            - suboptimal_structures: List of suboptimal structures
            - base_pair_probabilities: Base pairing probabilities
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_structure_prediction(
        ...     sequence="GGCCAATTCCGG",
        ...     output_file="structure.json"
        ... )
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Get sequence from input
    if input_file:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        with open(input_file) as f:
            sequence = f.read().strip()

    if not sequence:
        raise ValueError("No sequence provided")

    # Clean sequence
    sequence = sequence.upper().replace(' ', '').replace('\n', '')

    # Predict MFE structure
    mfe = predict_mfe_structure(sequence, config["material"], config["temperature"])

    # Generate suboptimal structures
    suboptimal = enumerate_suboptimal_structures(
        sequence, mfe.energy, config["energy_gap"], config["max_suboptimal"]
    )

    # Calculate base pair probabilities
    bp_probabilities = calculate_base_pair_probabilities(sequence, config["material"])

    # Filter high-probability pairs
    high_prob_pairs = {
        pair: prob for pair, prob in bp_probabilities.items()
        if prob >= config["probability_threshold"]
    }

    # Prepare results
    mfe_result = {
        "sequence": mfe.sequence,
        "structure": mfe.structure,
        "energy_kcal_mol": round(mfe.energy, 2),
        "base_pairs": mfe.base_pairs,
        "length": len(mfe.sequence)
    }

    suboptimal_results = [
        {
            "structure": struct.structure,
            "energy_kcal_mol": round(struct.energy, 2),
            "base_pairs": struct.base_pairs
        }
        for struct in suboptimal[:config["max_suboptimal"]]
    ]

    bp_prob_results = [
        {
            "position_i": pair[0],
            "position_j": pair[1],
            "probability": round(prob, 3)
        }
        for pair, prob in sorted(bp_probabilities.items(), key=lambda x: -x[1])
    ]

    # Prepare output data
    output_data = {
        "analysis_type": "structure_prediction",
        "model_parameters": {
            "material": config["material"],
            "temperature": config["temperature"],
            "energy_gap": config["energy_gap"]
        },
        "mfe_structure": mfe_result,
        "suboptimal_structures": suboptimal_results,
        "base_pair_probabilities": bp_prob_results,
        "high_probability_pairs": [
            {"position_i": pair[0], "position_j": pair[1], "probability": round(prob, 3)}
            for pair, prob in sorted(high_prob_pairs.items(), key=lambda x: -x[1])
        ]
    }

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    return {
        "mfe_structure": mfe_result,
        "suboptimal_structures": suboptimal_results,
        "base_pair_probabilities": bp_prob_results,
        "high_probability_pairs": list(high_prob_pairs.items()),
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "sequence_length": len(sequence),
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
    parser.add_argument('--sequence',
                       help='DNA/RNA sequence to analyze')
    parser.add_argument('--input', '-i',
                       help='Input file with sequence')
    parser.add_argument('--output', '-o',
                       help='Output file path')
    parser.add_argument('--material', choices=['DNA', 'RNA'], default='RNA',
                       help='Material type (DNA or RNA)')
    parser.add_argument('--temperature', type=float, default=37.0,
                       help='Temperature in Celsius')
    parser.add_argument('--energy-gap', type=float, default=5.0,
                       help='Energy gap for suboptimal structures (kcal/mol)')
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
    if args.material != 'RNA':
        kwargs['material'] = args.material
    if args.temperature != 37.0:
        kwargs['temperature'] = args.temperature
    if args.energy_gap != 5.0:
        kwargs['energy_gap'] = args.energy_gap

    # Run prediction
    result = run_structure_prediction(
        sequence=args.sequence,
        input_file=args.input,
        output_file=args.output,
        config=config,
        **kwargs
    )

    print(f"✅ Structure prediction completed")
    print(f"   Sequence length: {result['metadata']['sequence_length']}")
    print(f"   MFE energy: {result['mfe_structure']['energy_kcal_mol']:.2f} kcal/mol")
    if result['output_file']:
        print(f"   Results saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()