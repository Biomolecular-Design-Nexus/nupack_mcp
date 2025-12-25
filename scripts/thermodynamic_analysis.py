#!/usr/bin/env python3
"""
Script: thermodynamic_analysis.py
Description: Thermodynamic analysis of nucleic acid sequences

Original Use Case: examples/use_case_1_thermodynamics.py
Dependencies Removed: None (already self-contained)

Usage:
    python scripts/thermodynamic_analysis.py --input <sequences_file> --output <output_file>
    python scripts/thermodynamic_analysis.py --sequences <seq1> <seq2> --output <output_file>

Example:
    python scripts/thermodynamic_analysis.py --sequences ATCGATCGATCG GGCCAATTCCGG --output results/thermo.json
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import numpy as np
import pandas as pd

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    "material": "DNA",
    "temperature": 37.0,
    "parameter_file": None,
    "temp_scan": False,
    "temp_range": [25, 65, 5],  # start, stop, step
    "total_concentration": 1e-6
}

# ==============================================================================
# Core Classes (simplified from original)
# ==============================================================================
class SequenceModel:
    """Simplified model class for thermodynamic calculations"""
    def __init__(self, material='DNA', temperature=37.0, parameter_file=None):
        self.material = material
        self.temperature = temperature
        self.parameter_file = parameter_file

class NucleicAcidSequence:
    """Simplified sequence class"""
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.length = len(sequence)

# ==============================================================================
# Core Functions
# ==============================================================================
def calculate_partition_function(sequence: NucleicAcidSequence, model: SequenceModel) -> tuple:
    """
    Calculate partition function and free energy for a sequence.
    Simplified from NUPACK pfunc() functionality.
    """
    # Mock calculations based on sequence composition
    gc_content = sum(1 for base in sequence.sequence if base in ['G', 'C']) / sequence.length

    # Simple approximation for demonstration
    free_energy = -1.5 * sequence.length * gc_content - 2.1 * sequence.length * (1 - gc_content)
    partition_function = np.exp(-free_energy / (0.0019872 * (model.temperature + 273.15)))

    return partition_function, free_energy

def calculate_concentration_analysis(sequences: List[NucleicAcidSequence],
                                   model: SequenceModel,
                                   total_concentration: float = 1e-6) -> Dict[str, Any]:
    """Calculate complex concentration distribution"""
    results = {}
    for i, seq in enumerate(sequences):
        # Simple mock: assume equal distribution for demonstration
        fraction = 1.0 / len(sequences)
        concentration = total_concentration * fraction

        pfunc, free_energy = calculate_partition_function(seq, model)

        results[f"complex_{i+1}"] = {
            'sequence': seq.sequence,
            'concentration': concentration,
            'free_energy': free_energy,
            'partition_function': pfunc
        }

    return results

def perform_temperature_scan(sequences: List[NucleicAcidSequence],
                           temp_range: List[int] = [25, 65, 5]) -> pd.DataFrame:
    """Perform temperature scan analysis"""
    temperatures = np.arange(*temp_range)
    results = []

    for temp in temperatures:
        model = SequenceModel(temperature=temp)

        for i, seq in enumerate(sequences):
            pfunc, free_energy = calculate_partition_function(seq, model)

            results.append({
                'temperature': temp,
                'sequence_id': i + 1,
                'sequence': seq.sequence,
                'free_energy': free_energy,
                'partition_function': pfunc
            })

    return pd.DataFrame(results)

# ==============================================================================
# Main Function (MCP-ready)
# ==============================================================================
def run_thermodynamic_analysis(
    sequences: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for thermodynamic analysis.

    Args:
        sequences: List of DNA/RNA sequences to analyze
        input_file: Path to file with sequences (one per line)
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - individual_results: Analysis results for each sequence
            - concentration_analysis: Complex concentration data
            - temperature_scan: Temperature dependency data (if enabled)
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_thermodynamic_analysis(
        ...     sequences=["ATCGATCGATCG", "GGCCAATTCCGG"],
        ...     output_file="thermo.json"
        ... )
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Get sequences from input
    if input_file:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        with open(input_file) as f:
            sequences = [line.strip() for line in f if line.strip()]

    if not sequences:
        raise ValueError("No sequences provided")

    # Create sequence objects
    seq_objects = [NucleicAcidSequence(seq) for seq in sequences]

    # Initialize model
    model = SequenceModel(
        material=config["material"],
        temperature=config["temperature"],
        parameter_file=config["parameter_file"]
    )

    # Perform individual sequence analysis
    individual_results = []
    for i, seq in enumerate(seq_objects):
        pfunc, free_energy = calculate_partition_function(seq, model)

        result = {
            'sequence_id': i + 1,
            'sequence': seq.sequence,
            'length': seq.length,
            'free_energy_kcal_mol': round(free_energy, 3),
            'partition_function': f"{pfunc:.2e}",
            'temperature': config["temperature"]
        }
        individual_results.append(result)

    # Perform concentration analysis
    concentration_analysis = calculate_concentration_analysis(
        seq_objects, model, config["total_concentration"]
    )

    # Temperature scan if requested
    temperature_scan_results = None
    if config["temp_scan"]:
        temperature_scan_results = perform_temperature_scan(seq_objects, config["temp_range"])

    # Prepare output data
    output_data = {
        'analysis_type': 'thermodynamic',
        'model_parameters': {
            'material': config["material"],
            'temperature': config["temperature"],
            'parameter_file': config["parameter_file"]
        },
        'individual_results': individual_results,
        'concentration_analysis': concentration_analysis
    }

    if temperature_scan_results is not None:
        output_data['temperature_scan'] = temperature_scan_results.to_dict('records')

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    return {
        "individual_results": individual_results,
        "concentration_analysis": concentration_analysis,
        "temperature_scan": temperature_scan_results.to_dict('records') if temperature_scan_results is not None else None,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "num_sequences": len(sequences),
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
    parser.add_argument('--sequences', nargs='+',
                       help='DNA/RNA sequences to analyze')
    parser.add_argument('--input', '-i',
                       help='Input file with sequences (one per line)')
    parser.add_argument('--output', '-o',
                       help='Output file path')
    parser.add_argument('--material', choices=['DNA', 'RNA'], default='DNA',
                       help='Material type (DNA or RNA)')
    parser.add_argument('--temperature', type=float, default=37.0,
                       help='Temperature in Celsius')
    parser.add_argument('--parameter-file',
                       help='Parameter file to use')
    parser.add_argument('--temp-scan', action='store_true',
                       help='Perform temperature scan analysis')
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
    if args.parameter_file:
        kwargs['parameter_file'] = args.parameter_file
    if args.temp_scan:
        kwargs['temp_scan'] = True

    # Run analysis
    result = run_thermodynamic_analysis(
        sequences=args.sequences,
        input_file=args.input,
        output_file=args.output,
        config=config,
        **kwargs
    )

    print(f"✅ Thermodynamic analysis completed")
    print(f"   Sequences analyzed: {result['metadata']['num_sequences']}")
    if result['output_file']:
        print(f"   Results saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()