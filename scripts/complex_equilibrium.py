#!/usr/bin/env python3
"""
Script: complex_equilibrium.py
Description: Analyze complex equilibrium for multi-strand nucleic acid systems

Original Use Case: examples/use_case_4_complex_equilibrium.py
Dependencies Removed: matplotlib (now optional for plotting)

Usage:
    python scripts/complex_equilibrium.py --input <strands_file> --output <output_file>
    python scripts/complex_equilibrium.py --strands <strand1> <strand2> --output <output_file>

Example:
    python scripts/complex_equilibrium.py --strands ATCGATCGATCG CGAUCGAUCGAU --concentrations 1e-6 1e-6 --output results/equilibrium.json
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

# Optional plotting (graceful fallback if matplotlib unavailable)
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    "material": "DNA",
    "temperature": 37.0,
    "concentrations": [1e-6, 1e-6],  # M
    "max_complex_size": 2,
    "thermal_scan": False,
    "temp_range": [25, 75, 5],  # start, stop, step
    "stoichiometry_ratios": [0.1, 0.5, 1.0, 2.0, 10.0]
}

# ==============================================================================
# Core Classes
# ==============================================================================
class Strand:
    """Individual nucleic acid strand"""
    def __init__(self, sequence: str, name: str = None):
        self.sequence = sequence.upper()
        self.name = name or f"strand_{id(self)}"
        self.length = len(sequence)

class Complex:
    """Multi-strand complex"""
    def __init__(self, strands: List[Strand], stoichiometry: List[int]):
        self.strands = strands
        self.stoichiometry = stoichiometry
        self.name = self._generate_name()
        self.structure = self._predict_structure()
        self.free_energy = self._calculate_free_energy()

    def _generate_name(self) -> str:
        """Generate complex name from stoichiometry"""
        components = []
        for strand, count in zip(self.strands, self.stoichiometry):
            if count > 1:
                components.append(f"{strand.name}({count})")
            elif count == 1:
                components.append(strand.name)
        return "+".join(components)

    def _predict_structure(self) -> str:
        """Predict structure for multi-strand complex"""
        total_length = sum(s.length * count for s, count in zip(self.strands, self.stoichiometry))

        # Simple mock structure prediction
        if len(self.strands) == 1:
            # Single strand - can form hairpin
            sequence = self.strands[0].sequence
            structure = self._simple_hairpin_structure(sequence)
        else:
            # Multi-strand - assume duplex or complex interactions
            structure = self._multi_strand_structure()

        return structure

    def _simple_hairpin_structure(self, sequence: str) -> str:
        """Generate simple hairpin structure"""
        n = len(sequence)
        structure = ['.'] * n

        # Create simple hairpin in the middle
        if n >= 8:
            start = n // 4
            end = 3 * n // 4
            stem_length = min(3, (end - start) // 2)

            for i in range(stem_length):
                structure[start + i] = '('
                structure[end - i - 1] = ')'

        return ''.join(structure)

    def _multi_strand_structure(self) -> str:
        """Generate structure for multi-strand complex"""
        structures = []
        for strand, count in zip(self.strands, self.stoichiometry):
            for _ in range(count):
                # Simple duplex-like structure
                n = strand.length
                if len(structures) == 0:
                    # First strand - opening brackets
                    structures.append('(' * n)
                else:
                    # Subsequent strands - closing brackets
                    structures.append(')' * n)

        return '+'.join(structures)

    def _calculate_free_energy(self) -> float:
        """Calculate approximate free energy"""
        # Mock energy calculation based on sequence properties
        total_length = sum(s.length * count for s, count in zip(self.strands, self.stoichiometry))

        # Approximate energy based on composition and interactions
        energy = -2.1 * total_length  # Base stacking

        # Bonus for complementary interactions (mock)
        if len(self.strands) > 1:
            energy -= 5.0 * len(self.strands)  # Intermolecular interactions

        # Penalty for loop regions
        loops = self.structure.count('.')
        energy += 0.5 * loops

        return energy

# ==============================================================================
# Core Functions
# ==============================================================================
def enumerate_possible_complexes(strands: List[Strand], max_complex_size: int = 2) -> List[Complex]:
    """
    Enumerate all possible complexes from given strands.
    Simplified from NUPACK complex enumeration.
    """
    complexes = []

    # Single-strand complexes
    for strand in strands:
        complex_obj = Complex([strand], [1])
        complexes.append(complex_obj)

    # Multi-strand complexes
    if max_complex_size > 1:
        # Homo-dimers
        for strand in strands:
            if max_complex_size >= 2:
                complex_obj = Complex([strand], [2])
                complexes.append(complex_obj)

        # Hetero-dimers
        for i, strand1 in enumerate(strands):
            for j, strand2 in enumerate(strands[i+1:], i+1):
                complex_obj = Complex([strand1, strand2], [1, 1])
                complexes.append(complex_obj)

    return complexes

def calculate_equilibrium_concentrations(complexes: List[Complex],
                                       total_concentrations: List[float],
                                       temperature: float = 37.0) -> Dict[str, float]:
    """
    Calculate equilibrium concentrations of complexes.
    Simplified equilibrium calculation.
    """
    # Mock equilibrium calculation
    # In real NUPACK, this would solve mass balance equations

    equilibrium_conc = {}
    total_conc = sum(total_concentrations)

    # Simple partition based on stability (lower energy = higher concentration)
    energies = [complex_obj.free_energy for complex_obj in complexes]
    min_energy = min(energies)

    # Boltzmann-like distribution
    RT = 0.0019872 * (temperature + 273.15)  # kcal/mol
    partition_functions = [np.exp(-(e - min_energy) / RT) for e in energies]
    total_partition = sum(partition_functions)

    for complex_obj, pf in zip(complexes, partition_functions):
        fraction = pf / total_partition
        concentration = total_conc * fraction * 0.5  # Mock adjustment
        equilibrium_conc[complex_obj.name] = concentration

    return equilibrium_conc

def perform_thermal_analysis(strands: List[Strand],
                           concentrations: List[float],
                           temp_range: List[int] = [25, 75, 5]) -> pd.DataFrame:
    """Perform thermal stability analysis"""
    temperatures = np.arange(*temp_range)
    results = []

    for temp in temperatures:
        complexes = enumerate_possible_complexes(strands)
        eq_conc = calculate_equilibrium_concentrations(complexes, concentrations, temp)

        for complex_name, conc in eq_conc.items():
            # Find complex object
            complex_obj = next(c for c in complexes if c.name == complex_name)

            results.append({
                'temperature': temp,
                'complex': complex_name,
                'concentration': conc,
                'free_energy': complex_obj.free_energy,
                'fraction': conc / sum(eq_conc.values())
            })

    return pd.DataFrame(results)

def analyze_stoichiometry(strands: List[Strand],
                         ratios: List[float] = [0.1, 0.5, 1.0, 2.0, 10.0]) -> pd.DataFrame:
    """Analyze effect of stoichiometric ratios"""
    base_conc = 1e-6
    results = []

    for ratio in ratios:
        concentrations = [base_conc, base_conc * ratio]
        complexes = enumerate_possible_complexes(strands)
        eq_conc = calculate_equilibrium_concentrations(complexes, concentrations)

        # Find dominant complex
        dominant = max(eq_conc.items(), key=lambda x: x[1])

        results.append({
            'ratio': ratio,
            'strand_1_conc': concentrations[0],
            'strand_2_conc': concentrations[1],
            'dominant_complex': dominant[0],
            'dominant_concentration': dominant[1],
            'dominant_fraction': dominant[1] / sum(eq_conc.values())
        })

    return pd.DataFrame(results)

# ==============================================================================
# Main Function (MCP-ready)
# ==============================================================================
def run_complex_equilibrium(
    strands: Optional[List[str]] = None,
    input_file: Optional[Union[str, Path]] = None,
    concentrations: Optional[List[float]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for complex equilibrium analysis.

    Args:
        strands: List of DNA/RNA strand sequences
        input_file: Path to file with strand sequences (one per line)
        concentrations: Initial concentrations for each strand
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - complexes: List of possible complexes with properties
            - equilibrium_concentrations: Equilibrium concentrations
            - thermal_analysis: Temperature-dependent analysis (if enabled)
            - stoichiometry_analysis: Stoichiometric ratio analysis
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_complex_equilibrium(
        ...     strands=["ATCGATCGATCG", "CGAUCGAUCGAU"],
        ...     concentrations=[1e-6, 1e-6],
        ...     output_file="equilibrium.json"
        ... )
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Get strands from input
    if input_file:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        with open(input_file) as f:
            strands = [line.strip() for line in f if line.strip()]

    if not strands:
        raise ValueError("No strands provided")

    # Use provided concentrations or defaults
    if not concentrations:
        concentrations = config["concentrations"][:len(strands)]

    # Ensure concentrations match number of strands
    while len(concentrations) < len(strands):
        concentrations.append(concentrations[-1])  # Repeat last concentration
    concentrations = concentrations[:len(strands)]

    # Create strand objects
    strand_objects = [Strand(seq, f"strand_{i+1}") for i, seq in enumerate(strands)]

    # Enumerate possible complexes
    complexes = enumerate_possible_complexes(strand_objects, config["max_complex_size"])

    # Calculate equilibrium concentrations
    eq_concentrations = calculate_equilibrium_concentrations(
        complexes, concentrations, config["temperature"]
    )

    # Thermal analysis if requested
    thermal_results = None
    if config["thermal_scan"]:
        thermal_results = perform_thermal_analysis(strand_objects, concentrations, config["temp_range"])

    # Stoichiometry analysis (if multiple strands)
    stoich_results = None
    if len(strand_objects) >= 2:
        stoich_results = analyze_stoichiometry(strand_objects, config["stoichiometry_ratios"])

    # Prepare complex results
    complex_results = []
    for complex_obj in complexes:
        complex_results.append({
            "name": complex_obj.name,
            "strands": [s.sequence for s in complex_obj.strands],
            "stoichiometry": complex_obj.stoichiometry,
            "structure": complex_obj.structure,
            "free_energy_kcal_mol": round(complex_obj.free_energy, 2),
            "equilibrium_concentration": eq_concentrations.get(complex_obj.name, 0.0),
            "fraction": eq_concentrations.get(complex_obj.name, 0.0) / sum(eq_concentrations.values()) if eq_concentrations else 0.0
        })

    # Sort by concentration (descending)
    complex_results.sort(key=lambda x: x["equilibrium_concentration"], reverse=True)

    # Prepare output data
    output_data = {
        "analysis_type": "complex_equilibrium",
        "model_parameters": {
            "material": config["material"],
            "temperature": config["temperature"],
            "max_complex_size": config["max_complex_size"]
        },
        "input_strands": [
            {"sequence": seq, "concentration": conc}
            for seq, conc in zip(strands, concentrations)
        ],
        "complexes": complex_results,
        "equilibrium_concentrations": eq_concentrations
    }

    if thermal_results is not None:
        output_data["thermal_analysis"] = thermal_results.to_dict('records')

    if stoich_results is not None:
        output_data["stoichiometry_analysis"] = stoich_results.to_dict('records')

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    return {
        "complexes": complex_results,
        "equilibrium_concentrations": eq_concentrations,
        "thermal_analysis": thermal_results.to_dict('records') if thermal_results is not None else None,
        "stoichiometry_analysis": stoich_results.to_dict('records') if stoich_results is not None else None,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "num_strands": len(strands),
            "num_complexes": len(complexes),
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
    parser.add_argument('--strands', nargs='+',
                       help='DNA/RNA strand sequences')
    parser.add_argument('--input', '-i',
                       help='Input file with strand sequences (one per line)')
    parser.add_argument('--concentrations', nargs='+', type=float,
                       help='Initial concentrations for each strand')
    parser.add_argument('--output', '-o',
                       help='Output file path')
    parser.add_argument('--material', choices=['DNA', 'RNA'], default='DNA',
                       help='Material type (DNA or RNA)')
    parser.add_argument('--temperature', type=float, default=37.0,
                       help='Temperature in Celsius')
    parser.add_argument('--thermal-scan', action='store_true',
                       help='Perform thermal stability analysis')
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
    if args.thermal_scan:
        kwargs['thermal_scan'] = True

    # Run analysis
    result = run_complex_equilibrium(
        strands=args.strands,
        input_file=args.input,
        concentrations=args.concentrations,
        output_file=args.output,
        config=config,
        **kwargs
    )

    print(f"✅ Complex equilibrium analysis completed")
    print(f"   Strands analyzed: {result['metadata']['num_strands']}")
    print(f"   Complexes found: {result['metadata']['num_complexes']}")
    if result['output_file']:
        print(f"   Results saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()