#!/usr/bin/env python3
"""
NUPACK Use Case 5: Energy Evaluation
====================================

This script demonstrates energy evaluation capabilities of NUPACK:
- Structure energy calculations
- Energy landscape analysis
- Loop energy contributions
- Nearest-neighbor parameter analysis
- Structure comparison and ranking

This is a mock implementation that shows the typical NUPACK workflow.
For actual NUPACK functionality, the library must be properly compiled and installed.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Mock NUPACK classes for demonstration
class MockEnergyModel:
    """Mock energy model for structure evaluation"""
    def __init__(self, material='DNA', temperature=37.0, parameter_file=None):
        self.material = material
        self.temperature = temperature
        self.parameter_file = parameter_file
        self.RT = 0.0019872 * (temperature + 273.15)  # kcal/mol

    def structure_energy(self, sequence, structure):
        """Calculate structure energy using mock nearest-neighbor model"""
        energy = 0.0

        # Parse base pairs
        pairs = self._parse_structure(structure)

        # Stack energies (simplified nearest-neighbor)
        stack_energy = self._calculate_stack_energy(sequence, pairs)

        # Loop energies
        loop_energy = self._calculate_loop_energy(sequence, structure, pairs)

        # Terminal penalties
        terminal_penalty = self._calculate_terminal_penalty(sequence, pairs)

        total_energy = stack_energy + loop_energy + terminal_penalty

        return {
            'total_energy': total_energy,
            'stack_energy': stack_energy,
            'loop_energy': loop_energy,
            'terminal_penalty': terminal_penalty,
            'num_pairs': len(pairs)
        }

    def _parse_structure(self, structure):
        """Parse dot-bracket structure to get base pairs"""
        pairs = []
        stack = []

        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                pairs.append((j, i))

        return pairs

    def _calculate_stack_energy(self, sequence, pairs):
        """Calculate stacking energy contributions"""
        if len(pairs) < 2:
            return 0.0

        # Mock stacking parameters (kcal/mol)
        stack_params = {
            ('AT', 'AT'): -0.9, ('AT', 'TA'): -0.6, ('AT', 'CG'): -1.3, ('AT', 'GC'): -1.0,
            ('TA', 'AT'): -0.6, ('TA', 'TA'): -0.9, ('TA', 'CG'): -1.0, ('TA', 'GC'): -1.3,
            ('CG', 'AT'): -1.3, ('CG', 'TA'): -1.0, ('CG', 'CG'): -2.1, ('CG', 'GC'): -1.4,
            ('GC', 'AT'): -1.0, ('GC', 'TA'): -1.3, ('GC', 'CG'): -1.4, ('GC', 'GC'): -2.1
        }

        # Convert to RNA if needed
        if 'U' in sequence:
            sequence = sequence.replace('T', 'U')
            # Update parameters for RNA (slightly different)
            for key in list(stack_params.keys()):
                new_key = (key[0].replace('T', 'U'), key[1].replace('T', 'U'))
                stack_params[new_key] = stack_params[key] - 0.1  # RNA slightly less stable

        total_stack = 0.0
        sorted_pairs = sorted(pairs)

        for i in range(len(sorted_pairs) - 1):
            i1, j1 = sorted_pairs[i]
            i2, j2 = sorted_pairs[i + 1]

            # Check if pairs are adjacent (stackable)
            if i2 == i1 + 1 and j2 == j1 - 1:
                # Adjacent pairs - calculate stack energy
                bp1 = sequence[i1] + sequence[j1]
                bp2 = sequence[i2] + sequence[j2]

                # Reverse complement for proper stacking geometry
                complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'U': 'A'}
                if 'U' in sequence:
                    complement['A'] = 'U'

                if bp1 in stack_params and bp2 in stack_params:
                    stack_key = (bp1, bp2)
                    if stack_key in stack_params:
                        total_stack += stack_params[stack_key]
                    else:
                        total_stack -= 1.0  # Default unfavorable

        return total_stack

    def _calculate_loop_energy(self, sequence, structure, pairs):
        """Calculate loop energy contributions"""
        total_loop_energy = 0.0

        # Find loops
        loops = self._find_loops(structure)

        for loop in loops:
            loop_size = len(loop)

            if loop_size <= 3:
                # Very small loops are highly destabilized
                total_loop_energy += 5.0 + loop_size * 0.5
            elif loop_size <= 6:
                # Small loops (hairpins)
                total_loop_energy += 3.0 + 0.3 * loop_size
            elif loop_size <= 30:
                # Medium loops
                total_loop_energy += 1.75 * np.log(loop_size)
            else:
                # Large loops
                total_loop_energy += 1.75 * np.log(30) + 0.1 * (loop_size - 30)

        return total_loop_energy

    def _find_loops(self, structure):
        """Identify loop regions in structure"""
        loops = []
        in_loop = False
        current_loop = []

        for i, char in enumerate(structure):
            if char == '.':
                if not in_loop:
                    in_loop = True
                    current_loop = [i]
                else:
                    current_loop.append(i)
            else:
                if in_loop:
                    loops.append(current_loop)
                    current_loop = []
                    in_loop = False

        if in_loop:
            loops.append(current_loop)

        return loops

    def _calculate_terminal_penalty(self, sequence, pairs):
        """Calculate terminal AU penalty for RNA or terminal AT penalty for DNA"""
        if not pairs:
            return 0.0

        penalty = 0.0

        # Check terminal pairs
        for i, j in pairs:
            # Check if this is a terminal pair (no other pairs enclosing it)
            is_terminal = True
            for ii, jj in pairs:
                if ii < i and jj > j:
                    is_terminal = False
                    break

            if is_terminal:
                bp = sequence[i] + sequence[j]
                if bp in ['AT', 'TA', 'AU', 'UA']:
                    penalty += 0.5  # Terminal AU/AT penalty

        return penalty

def compare_structures(sequence, structures, energy_model):
    """Compare energy of different structures for the same sequence"""
    results = []

    for i, structure in enumerate(structures):
        if len(sequence) != len(structure):
            print(f"Warning: Sequence length ({len(sequence)}) doesn't match structure {i+1} length ({len(structure)})")
            continue

        energy_data = energy_model.structure_energy(sequence, structure)

        result = {
            'structure_id': i + 1,
            'structure': structure,
            'total_energy': energy_data['total_energy'],
            'stack_energy': energy_data['stack_energy'],
            'loop_energy': energy_data['loop_energy'],
            'terminal_penalty': energy_data['terminal_penalty'],
            'num_pairs': energy_data['num_pairs'],
            'pairs_per_nt': energy_data['num_pairs'] / len(sequence)
        }

        results.append(result)

    # Sort by energy (most favorable first)
    return sorted(results, key=lambda x: x['total_energy'])

def energy_landscape_analysis(sequence, energy_model, num_structures=20):
    """Generate random structures and analyze energy landscape"""
    import random

    structures = []

    # Generate MFE-like structure
    mfe_structure = generate_mock_mfe_structure(sequence)
    structures.append(mfe_structure)

    # Generate random variations
    for _ in range(num_structures - 1):
        variant = generate_structure_variant(mfe_structure)
        structures.append(variant)

    # Evaluate all structures
    results = compare_structures(sequence, structures, energy_model)

    return results

def generate_mock_mfe_structure(sequence):
    """Generate a mock MFE structure"""
    n = len(sequence)
    structure = ['.'] * n

    # Simple algorithm: find complementary regions and pair them
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'U': 'A'}
    if 'U' in sequence:
        complement['A'] = 'U'

    for i in range(n - 4):  # Minimum loop size
        for j in range(i + 4, n):
            if (structure[i] == '.' and structure[j] == '.' and
                sequence[i] in complement and sequence[j] == complement[sequence[i]]):
                structure[i] = '('
                structure[j] = ')'
                break

    return ''.join(structure)

def generate_structure_variant(base_structure):
    """Generate a variant of a structure by changing some base pairs"""
    import random

    structure = list(base_structure)
    n = len(structure)

    # Find existing base pairs
    pairs = []
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            pairs.append((j, i))

    # Randomly remove some pairs
    if pairs:
        num_to_remove = random.randint(0, min(2, len(pairs)))
        pairs_to_remove = random.sample(pairs, num_to_remove)

        for i, j in pairs_to_remove:
            structure[i] = '.'
            structure[j] = '.'

    return ''.join(structure)

def parameter_sensitivity_analysis(sequence, structure, energy_model):
    """Analyze sensitivity to parameter changes"""
    base_energy = energy_model.structure_energy(sequence, structure)

    # Test different temperatures
    temperatures = [25, 37, 50, 65, 80]
    temp_results = []

    for temp in temperatures:
        temp_model = MockEnergyModel(
            material=energy_model.material,
            temperature=temp,
            parameter_file=energy_model.parameter_file
        )
        energy_data = temp_model.structure_energy(sequence, structure)
        temp_results.append({
            'temperature': temp,
            'total_energy': energy_data['total_energy'],
            'stack_energy': energy_data['stack_energy'],
            'loop_energy': energy_data['loop_energy']
        })

    return {
        'base_energy': base_energy,
        'temperature_sensitivity': temp_results
    }

def main():
    parser = argparse.ArgumentParser(description='NUPACK Energy Evaluation')
    parser.add_argument('--sequence', type=str, default='GGGAAACCC',
                       help='RNA/DNA sequence')
    parser.add_argument('--structure', type=str,
                       help='Structure in dot-bracket notation')
    parser.add_argument('--structures', nargs='+',
                       help='Multiple structures to compare')
    parser.add_argument('--structures-file', type=str,
                       help='File containing structures (one per line)')
    parser.add_argument('--material', choices=['DNA', 'RNA'], default='RNA',
                       help='Material type')
    parser.add_argument('--temperature', type=float, default=37.0,
                       help='Temperature in Celsius')
    parser.add_argument('--parameter-file', type=str,
                       default='examples/data/parameters/rna99.json',
                       help='Parameter file')
    parser.add_argument('--landscape', action='store_true',
                       help='Perform energy landscape analysis')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Perform parameter sensitivity analysis')
    parser.add_argument('--output', type=str,
                       help='Output file for results')

    args = parser.parse_args()

    # Handle input structures
    structures = []
    if args.structures:
        structures = args.structures
    elif args.structures_file:
        try:
            with open(args.structures_file, 'r') as f:
                structures = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File {args.structures_file} not found")
            return
    elif args.structure:
        structures = [args.structure]
    else:
        # Generate default structure
        structures = [generate_mock_mfe_structure(args.sequence)]

    # Initialize energy model
    param_file = Path(args.parameter_file)
    if not param_file.exists():
        print(f"Warning: Parameter file {param_file} not found")
        print("Using default parameters")
        args.parameter_file = None

    energy_model = MockEnergyModel(
        material=args.material,
        temperature=args.temperature,
        parameter_file=args.parameter_file
    )

    print(f"Initialized {args.material} energy model at {args.temperature}°C")
    if args.parameter_file:
        print(f"Using parameters: {args.parameter_file}")

    print("\n" + "="*60)
    print("NUPACK ENERGY EVALUATION ANALYSIS")
    print("="*60)

    sequence = args.sequence.upper()
    print(f"\nSequence: {sequence}")
    print(f"Length: {len(sequence)} nucleotides")

    all_results = {
        'analysis_type': 'energy_evaluation',
        'sequence': sequence,
        'model_parameters': {
            'material': args.material,
            'temperature': args.temperature,
            'parameter_file': args.parameter_file
        }
    }

    # Structure comparison
    print(f"\n1. STRUCTURE ENERGY EVALUATION")
    print("-" * 40)

    comparison_results = compare_structures(sequence, structures, energy_model)

    print(f"Evaluated {len(comparison_results)} structures:")
    print("\nRanked by energy (most favorable first):")

    for i, result in enumerate(comparison_results):
        print(f"\nRank {i+1}: Structure {result['structure_id']}")
        print(f"  Structure: {result['structure']}")
        print(f"  Total Energy: {result['total_energy']:.2f} kcal/mol")
        print(f"  Stack Energy: {result['stack_energy']:.2f} kcal/mol")
        print(f"  Loop Energy: {result['loop_energy']:.2f} kcal/mol")
        print(f"  Terminal Penalty: {result['terminal_penalty']:.2f} kcal/mol")
        print(f"  Base Pairs: {result['num_pairs']} ({result['pairs_per_nt']:.1%})")

    all_results['structure_comparison'] = comparison_results

    # Energy landscape analysis
    if args.landscape:
        print(f"\n2. ENERGY LANDSCAPE ANALYSIS")
        print("-" * 40)

        landscape_results = energy_landscape_analysis(sequence, energy_model)

        print(f"Generated and evaluated {len(landscape_results)} structures")

        # Energy distribution statistics
        energies = [r['total_energy'] for r in landscape_results]
        print(f"\nEnergy landscape statistics:")
        print(f"  Mean: {np.mean(energies):.2f} kcal/mol")
        print(f"  Std:  {np.std(energies):.2f} kcal/mol")
        print(f"  Min:  {np.min(energies):.2f} kcal/mol (MFE)")
        print(f"  Max:  {np.max(energies):.2f} kcal/mol")
        print(f"  Range: {np.max(energies) - np.min(energies):.2f} kcal/mol")

        # Show top 5 structures
        print(f"\nTop 5 most stable structures:")
        for i, result in enumerate(landscape_results[:5]):
            print(f"  {i+1}. Energy: {result['total_energy']:.2f} kcal/mol, Pairs: {result['num_pairs']}")

        all_results['energy_landscape'] = {
            'structures': landscape_results,
            'statistics': {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies)),
                'range': float(np.max(energies) - np.min(energies))
            }
        }

    # Parameter sensitivity analysis
    if args.sensitivity and structures:
        print(f"\n3. PARAMETER SENSITIVITY ANALYSIS")
        print("-" * 40)

        structure_for_sensitivity = structures[0]
        sensitivity_results = parameter_sensitivity_analysis(
            sequence, structure_for_sensitivity, energy_model)

        print(f"Structure analyzed: {structure_for_sensitivity}")

        print(f"\nBase energy components:")
        base = sensitivity_results['base_energy']
        print(f"  Total: {base['total_energy']:.2f} kcal/mol")
        print(f"  Stack: {base['stack_energy']:.2f} kcal/mol")
        print(f"  Loop:  {base['loop_energy']:.2f} kcal/mol")

        print(f"\nTemperature sensitivity:")
        for temp_data in sensitivity_results['temperature_sensitivity']:
            print(f"  {temp_data['temperature']}°C: {temp_data['total_energy']:.2f} kcal/mol")

        all_results['sensitivity_analysis'] = sensitivity_results

    # Energy decomposition for best structure
    if comparison_results:
        print(f"\n4. DETAILED ENERGY DECOMPOSITION")
        print("-" * 40)

        best_structure = comparison_results[0]
        print(f"Most favorable structure: {best_structure['structure']}")

        # Component analysis
        total = best_structure['total_energy']
        stack = best_structure['stack_energy']
        loop = best_structure['loop_energy']
        terminal = best_structure['terminal_penalty']

        print(f"\nEnergy breakdown:")
        print(f"  Stack energy:      {stack:8.2f} kcal/mol ({stack/total*100:5.1f}%)")
        print(f"  Loop energy:       {loop:8.2f} kcal/mol ({loop/total*100:5.1f}%)")
        print(f"  Terminal penalty:  {terminal:8.2f} kcal/mol ({terminal/total*100:5.1f}%)")
        print(f"  Total:            {total:8.2f} kcal/mol")

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "="*60)
    print("Energy evaluation completed!")
    print("Note: This is a mock implementation for MCP development.")
    print("For actual NUPACK functionality, install the compiled NUPACK library.")

if __name__ == '__main__':
    main()