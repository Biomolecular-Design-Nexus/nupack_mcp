#!/usr/bin/env python3
"""
Script: energy_evaluation.py
Description: Evaluate energy landscape and properties of nucleic acid structures

Original Use Case: examples/use_case_5_energy_evaluation.py
Dependencies Removed: None (already self-contained)

Usage:
    python scripts/energy_evaluation.py --input <sequence_file> --output <output_file>
    python scripts/energy_evaluation.py --sequence <sequence> --structure <structure> --output <output_file>

Example:
    python scripts/energy_evaluation.py --sequence GGCCAATTCCGG --structure "((((....))))" --output results/energy.json
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
    "parameter_file": None,
    "landscape_analysis": False,
    "landscape_size": 20,
    "sensitivity_analysis": False,
    "temp_range": [25, 80, 5]
}

# ==============================================================================
# Core Classes
# ==============================================================================
class EnergyModel:
    """Energy model for nucleic acid structures"""
    def __init__(self, material: str = "RNA", temperature: float = 37.0, parameter_file: str = None):
        self.material = material
        self.temperature = temperature
        self.parameter_file = parameter_file
        self.parameters = self._load_parameters()

    def _load_parameters(self) -> Dict[str, float]:
        """Load thermodynamic parameters"""
        if self.parameter_file and Path(self.parameter_file).exists():
            try:
                with open(self.parameter_file) as f:
                    return json.load(f)
            except:
                pass

        # Default mock parameters
        return {
            "stack_energy_gc": -3.0,
            "stack_energy_au": -1.5,
            "loop_penalty": 4.2,
            "terminal_penalty": 0.8,
            "mismatch_penalty": 2.5
        }

class StructureEnergy:
    """Container for structure energy evaluation"""
    def __init__(self, sequence: str, structure: str, energy: float):
        self.sequence = sequence
        self.structure = structure
        self.energy = energy
        self.base_pairs = self._extract_base_pairs()
        self.energy_components = self._decompose_energy()

    def _extract_base_pairs(self) -> List[Tuple[int, int]]:
        """Extract base pairs from structure"""
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

    def _decompose_energy(self) -> Dict[str, float]:
        """Decompose energy into components"""
        # Mock energy decomposition
        stack_energy = len(self.base_pairs) * -2.0
        loop_energy = self.structure.count('.') * 0.5
        terminal_energy = 0.8 if self.base_pairs else 0.0

        return {
            "stack_energy": stack_energy,
            "loop_energy": loop_energy,
            "terminal_energy": terminal_energy,
            "total_energy": stack_energy + loop_energy + terminal_energy
        }

# ==============================================================================
# Core Functions
# ==============================================================================
def calculate_structure_energy(sequence: str, structure: str, model: EnergyModel) -> StructureEnergy:
    """
    Calculate energy for a given sequence-structure pair.
    Simplified from NUPACK energy() functionality.
    """
    # Validate inputs
    if len(sequence) != len(structure):
        raise ValueError("Sequence and structure must have same length")

    # Extract base pairs
    base_pairs = []
    stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                base_pairs.append((j, i))

    # Calculate energy components
    stack_energy = 0.0
    for i, j in base_pairs:
        base_i = sequence[i]
        base_j = sequence[j]

        # GC pairs are more stable
        if (base_i in ['G', 'C'] and base_j in ['G', 'C']):
            stack_energy += model.parameters["stack_energy_gc"]
        else:
            stack_energy += model.parameters["stack_energy_au"]

    # Loop penalties
    loop_energy = structure.count('.') * model.parameters["loop_penalty"] / 10

    # Terminal penalties
    terminal_energy = model.parameters["terminal_penalty"] if base_pairs else 0.0

    total_energy = stack_energy + loop_energy + terminal_energy

    return StructureEnergy(sequence, structure, total_energy)

def generate_energy_landscape(sequence: str, model: EnergyModel, num_structures: int = 20) -> List[StructureEnergy]:
    """
    Generate energy landscape by sampling alternative structures.
    Simplified from suboptimal structure enumeration.
    """
    structures = []

    # Start with MFE structure prediction
    mfe_structure = predict_mfe_structure(sequence)
    mfe_energy = calculate_structure_energy(sequence, mfe_structure, model)
    structures.append(mfe_energy)

    # Generate alternative structures
    for i in range(1, num_structures):
        alt_structure = generate_alternative_structure(sequence, mfe_structure, i)
        alt_energy = calculate_structure_energy(sequence, alt_structure, model)
        structures.append(alt_energy)

    return sorted(structures, key=lambda s: s.energy)

def predict_mfe_structure(sequence: str) -> str:
    """Simple MFE structure prediction"""
    n = len(sequence)
    structure = ['.'] * n

    # Simple base pairing algorithm
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

    for i in range(n - 3):
        for j in range(i + 4, n):
            if sequence[i] in complement and sequence[j] == complement[sequence[i]]:
                if structure[i] == '.' and structure[j] == '.':
                    structure[i] = '('
                    structure[j] = ')'
                    break

    return ''.join(structure)

def generate_alternative_structure(sequence: str, base_structure: str, variant: int) -> str:
    """Generate alternative structure by modifying base structure"""
    n = len(sequence)
    structure = list(base_structure)

    # Remove some base pairs to create alternatives
    pairs = []
    stack = []

    for i, char in enumerate(base_structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))

    # Remove some pairs based on variant number
    remove_count = min(variant // 3, len(pairs))
    for k in range(remove_count):
        if k < len(pairs):
            i, j = pairs[k]
            structure[i] = '.'
            structure[j] = '.'

    return ''.join(structure)

def perform_sensitivity_analysis(sequence: str, structure: str, base_model: EnergyModel) -> Dict[str, Any]:
    """
    Analyze sensitivity to parameter changes.
    """
    base_energy = calculate_structure_energy(sequence, structure, base_model)
    sensitivities = {}

    # Test parameter variations
    parameter_variations = {
        "stack_energy_gc": [-0.5, 0.5],
        "stack_energy_au": [-0.3, 0.3],
        "loop_penalty": [-0.2, 0.2],
        "terminal_penalty": [-0.1, 0.1]
    }

    for param, variations in parameter_variations.items():
        param_sensitivity = []

        for delta in variations:
            # Create modified model
            modified_params = base_model.parameters.copy()
            modified_params[param] += delta
            modified_model = EnergyModel(base_model.material, base_model.temperature)
            modified_model.parameters = modified_params

            # Calculate energy with modified parameters
            modified_energy = calculate_structure_energy(sequence, structure, modified_model)
            energy_change = modified_energy.energy - base_energy.energy

            param_sensitivity.append({
                "parameter_change": delta,
                "energy_change": energy_change,
                "sensitivity": energy_change / delta if delta != 0 else 0.0
            })

        sensitivities[param] = param_sensitivity

    return sensitivities

def analyze_temperature_dependence(sequence: str, structure: str,
                                 temp_range: List[int] = [25, 80, 5]) -> pd.DataFrame:
    """Analyze energy temperature dependence"""
    temperatures = np.arange(*temp_range)
    results = []

    for temp in temperatures:
        model = EnergyModel(temperature=temp)
        energy = calculate_structure_energy(sequence, structure, model)

        results.append({
            'temperature': temp,
            'energy': energy.energy,
            'stack_energy': energy.energy_components['stack_energy'],
            'loop_energy': energy.energy_components['loop_energy'],
            'terminal_energy': energy.energy_components['terminal_energy']
        })

    return pd.DataFrame(results)

# ==============================================================================
# Main Function (MCP-ready)
# ==============================================================================
def run_energy_evaluation(
    sequence: Optional[str] = None,
    structure: Optional[str] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for energy evaluation.

    Args:
        sequence: DNA/RNA sequence
        structure: Secondary structure in dot-bracket notation
        input_file: Path to file with sequence and structure
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - structure_energy: Energy evaluation for given structure
            - landscape_analysis: Energy landscape (if enabled)
            - sensitivity_analysis: Parameter sensitivity (if enabled)
            - temperature_analysis: Temperature dependence
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_energy_evaluation(
        ...     sequence="GGCCAATTCCGG",
        ...     structure="((((....))))",
        ...     output_file="energy.json"
        ... )
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Get sequence and structure from input
    if input_file:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file) as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) >= 1:
                sequence = lines[0]
            if len(lines) >= 2:
                structure = lines[1]

    if not sequence:
        raise ValueError("No sequence provided")

    if not structure:
        # Predict structure if not provided
        structure = predict_mfe_structure(sequence)

    # Validate inputs
    if len(sequence) != len(structure):
        raise ValueError("Sequence and structure must have same length")

    # Initialize energy model
    model = EnergyModel(
        material=config["material"],
        temperature=config["temperature"],
        parameter_file=config["parameter_file"]
    )

    # Calculate structure energy
    structure_energy = calculate_structure_energy(sequence, structure, model)

    # Energy landscape analysis
    landscape_results = None
    if config["landscape_analysis"]:
        landscape = generate_energy_landscape(sequence, model, config["landscape_size"])
        landscape_results = [
            {
                "structure": s.structure,
                "energy": round(s.energy, 2),
                "energy_components": {k: round(v, 2) for k, v in s.energy_components.items()}
            }
            for s in landscape
        ]

    # Sensitivity analysis
    sensitivity_results = None
    if config["sensitivity_analysis"]:
        sensitivity_results = perform_sensitivity_analysis(sequence, structure, model)

    # Temperature analysis
    temp_analysis = analyze_temperature_dependence(sequence, structure, config["temp_range"])

    # Prepare structure energy result
    energy_result = {
        "sequence": structure_energy.sequence,
        "structure": structure_energy.structure,
        "total_energy_kcal_mol": round(structure_energy.energy, 2),
        "energy_components": {k: round(v, 2) for k, v in structure_energy.energy_components.items()},
        "base_pairs": structure_energy.base_pairs,
        "length": len(sequence)
    }

    # Prepare output data
    output_data = {
        "analysis_type": "energy_evaluation",
        "model_parameters": {
            "material": config["material"],
            "temperature": config["temperature"],
            "parameter_file": config["parameter_file"]
        },
        "structure_energy": energy_result,
        "temperature_analysis": temp_analysis.to_dict('records')
    }

    if landscape_results:
        output_data["landscape_analysis"] = {
            "structures": landscape_results,
            "statistics": {
                "mean_energy": np.mean([s["energy"] for s in landscape_results]),
                "min_energy": min([s["energy"] for s in landscape_results]),
                "max_energy": max([s["energy"] for s in landscape_results]),
                "energy_range": max([s["energy"] for s in landscape_results]) - min([s["energy"] for s in landscape_results])
            }
        }

    if sensitivity_results:
        output_data["sensitivity_analysis"] = sensitivity_results

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    return {
        "structure_energy": energy_result,
        "landscape_analysis": landscape_results,
        "sensitivity_analysis": sensitivity_results,
        "temperature_analysis": temp_analysis.to_dict('records'),
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "sequence_length": len(sequence),
            "num_base_pairs": len(structure_energy.base_pairs),
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
                       help='DNA/RNA sequence')
    parser.add_argument('--structure',
                       help='Structure in dot-bracket notation')
    parser.add_argument('--input', '-i',
                       help='Input file with sequence and structure')
    parser.add_argument('--output', '-o',
                       help='Output file path')
    parser.add_argument('--material', choices=['DNA', 'RNA'], default='RNA',
                       help='Material type (DNA or RNA)')
    parser.add_argument('--temperature', type=float, default=37.0,
                       help='Temperature in Celsius')
    parser.add_argument('--parameter-file',
                       help='Parameter file to use')
    parser.add_argument('--landscape', action='store_true',
                       help='Perform energy landscape analysis')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Perform parameter sensitivity analysis')
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
    if args.parameter_file:
        kwargs['parameter_file'] = args.parameter_file
    if args.landscape:
        kwargs['landscape_analysis'] = True
    if args.sensitivity:
        kwargs['sensitivity_analysis'] = True

    # Run evaluation
    result = run_energy_evaluation(
        sequence=args.sequence,
        structure=args.structure,
        input_file=args.input,
        output_file=args.output,
        config=config,
        **kwargs
    )

    print(f"✅ Energy evaluation completed")
    print(f"   Sequence length: {result['metadata']['sequence_length']}")
    print(f"   Base pairs: {result['metadata']['num_base_pairs']}")
    print(f"   Total energy: {result['structure_energy']['total_energy_kcal_mol']:.2f} kcal/mol")
    if result['output_file']:
        print(f"   Results saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()