#!/usr/bin/env python3
"""
NUPACK Use Case 4: Complex Equilibrium Analysis
===============================================

This script demonstrates complex equilibrium analysis capabilities of NUPACK:
- Multi-complex equilibrium concentration calculations
- Test tube analysis with multiple strands
- Competition analysis between different complexes
- Thermal stability analysis

This is a mock implementation that shows the typical NUPACK workflow.
For actual NUPACK functionality, the library must be properly compiled and installed.
"""

import argparse
import json
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt

# Mock NUPACK classes for demonstration
class MockStrand:
    """Mock NUPACK Strand class"""
    def __init__(self, sequence, name=None):
        self.sequence = sequence.upper()
        self.name = name or f"strand_{len(sequence)}"
        self.length = len(sequence)

    def __str__(self):
        return f"{self.name}: {self.sequence}"

class MockComplex:
    """Mock NUPACK Complex class"""
    def __init__(self, strands, structure=None):
        self.strands = strands if isinstance(strands, list) else [strands]
        self.sequence = '+'.join(s.sequence for s in self.strands)
        self.structure = structure or self._predict_structure()
        self.free_energy = self._calculate_free_energy()

    def _predict_structure(self):
        """Mock structure prediction for complex"""
        total_length = sum(s.length for s in self.strands)
        # Simplified: assume some base pairing
        if len(self.strands) == 1:
            # Single strand - can form hairpin
            return self._hairpin_structure(self.strands[0].sequence)
        else:
            # Multiple strands - assume some intermolecular pairing
            return self._duplex_structure()

    def _hairpin_structure(self, sequence):
        """Simple hairpin structure prediction"""
        n = len(sequence)
        if n < 6:
            return '.' * n

        # Simple stem-loop: first quarter pairs with last quarter
        stem_length = min(n // 4, 3)
        structure = ['('] * stem_length + ['.'] * (n - 2 * stem_length) + [')'] * stem_length
        return ''.join(structure)

    def _duplex_structure(self):
        """Simple duplex structure for multiple strands"""
        lengths = [s.length for s in self.strands]
        structures = []

        for i, length in enumerate(lengths):
            if i == 0:
                # First strand mostly paired
                paired = min(length, sum(lengths[1:]))
                structures.append('(' * paired + '.' * (length - paired))
            else:
                # Other strands complement first
                structures.append(')' * length)

        return '+'.join(structures)

    def _calculate_free_energy(self):
        """Mock free energy calculation"""
        bp_count = self.structure.count('(')
        total_length = sum(s.length for s in self.strands)

        # Simple approximation: paired bases contribute favorably
        pairing_energy = -2.1 * bp_count  # kcal/mol per bp
        entropy_penalty = 0.1 * total_length  # entropy cost

        # Intermolecular penalty for multiple strands
        if len(self.strands) > 1:
            association_penalty = 1.5 * (len(self.strands) - 1)
            return pairing_energy + entropy_penalty + association_penalty

        return pairing_energy + entropy_penalty

    def __str__(self):
        return f"Complex({'+'.join(s.name for s in self.strands)}, ΔG={self.free_energy:.2f})"

class MockTube:
    """Mock NUPACK Tube class for equilibrium analysis"""
    def __init__(self, strands, concentrations=None):
        self.strands = strands
        self.concentrations = concentrations or [1e-6] * len(strands)  # Default 1 μM
        self.complexes = self._enumerate_complexes()
        self.equilibrium_concentrations = self._calculate_equilibrium()

    def _enumerate_complexes(self):
        """Enumerate possible complexes from strands"""
        complexes = []

        # Single-strand complexes (monomers)
        for strand in self.strands:
            complexes.append(MockComplex([strand]))

        # Two-strand complexes (dimers)
        for i, strand1 in enumerate(self.strands):
            for j, strand2 in enumerate(self.strands):
                if i <= j:  # Avoid duplicates
                    if i == j:
                        # Homodimer
                        complexes.append(MockComplex([strand1, strand1]))
                    else:
                        # Heterodimer
                        complexes.append(MockComplex([strand1, strand2]))

        return complexes

    def _calculate_equilibrium(self):
        """Mock equilibrium concentration calculation"""
        # Simple approximation using Boltzmann factors
        RT = 0.0019872 * (273.15 + 37)  # kcal/mol at 37°C

        # Calculate relative probabilities
        probabilities = []
        for complex in self.complexes:
            # Boltzmann factor
            prob = np.exp(-complex.free_energy / RT)
            probabilities.append(prob)

        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        # Approximate concentrations based on total strand concentrations
        total_conc = sum(self.concentrations)
        concentrations = [p * total_conc for p in probabilities]

        return concentrations

    def get_complex_fractions(self):
        """Get fraction of each complex at equilibrium"""
        total_conc = sum(self.equilibrium_concentrations)
        if total_conc == 0:
            return [0] * len(self.complexes)
        return [c / total_conc for c in self.equilibrium_concentrations]

class MockModel:
    """Mock NUPACK Model class"""
    def __init__(self, material='DNA', temperature=37.0):
        self.material = material
        self.temperature = temperature

def analyze_competition(tubes, model):
    """Analyze competition between different tube compositions"""
    results = []

    for i, tube in enumerate(tubes):
        fractions = tube.get_complex_fractions()

        tube_result = {
            'tube_id': i + 1,
            'strand_concentrations': tube.concentrations,
            'complexes': [],
            'dominant_complex': None
        }

        max_fraction = 0
        for j, (complex, conc, fraction) in enumerate(zip(tube.complexes, tube.equilibrium_concentrations, fractions)):
            complex_info = {
                'complex_id': j + 1,
                'strands': [s.name for s in complex.strands],
                'concentration': conc,
                'fraction': fraction,
                'free_energy': complex.free_energy
            }
            tube_result['complexes'].append(complex_info)

            if fraction > max_fraction:
                max_fraction = fraction
                tube_result['dominant_complex'] = complex_info

        results.append(tube_result)

    return results

def thermal_stability_analysis(tube, temperature_range=(25, 75, 5)):
    """Analyze thermal stability of complexes"""
    temperatures = np.arange(*temperature_range)
    results = []

    for temp in temperatures:
        # Recalculate equilibrium at different temperature
        RT = 0.0019872 * (temp + 273.15)

        probabilities = []
        for complex in tube.complexes:
            prob = np.exp(-complex.free_energy / RT)
            probabilities.append(prob)

        total_prob = sum(probabilities)
        fractions = [p / total_prob for p in probabilities]

        for i, (complex, fraction) in enumerate(zip(tube.complexes, fractions)):
            results.append({
                'temperature': temp,
                'complex_id': i,
                'complex_name': '+'.join(s.name for s in complex.strands),
                'fraction': fraction,
                'free_energy': complex.free_energy
            })

    return pd.DataFrame(results)

def titration_analysis(strand_a, strand_b, concentrations_b, fixed_conc_a=1e-6):
    """Simulate titration experiment"""
    results = []

    for conc_b in concentrations_b:
        tube = MockTube([strand_a, strand_b], [fixed_conc_a, conc_b])
        fractions = tube.get_complex_fractions()

        # Find heterodimer complex
        heterodimer_idx = None
        for i, complex in enumerate(tube.complexes):
            if len(complex.strands) == 2 and complex.strands[0].name != complex.strands[1].name:
                heterodimer_idx = i
                break

        heterodimer_fraction = fractions[heterodimer_idx] if heterodimer_idx is not None else 0

        results.append({
            'strand_b_concentration': conc_b,
            'strand_a_concentration': fixed_conc_a,
            'ratio_b_to_a': conc_b / fixed_conc_a,
            'heterodimer_fraction': heterodimer_fraction,
            'free_strand_a_fraction': fractions[0] if len(fractions) > 0 else 0
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='NUPACK Complex Equilibrium Analysis')
    parser.add_argument('--strands', nargs='+',
                       default=['ATCGATCGATCG', 'CGATCGATCGAT'],
                       help='Strand sequences')
    parser.add_argument('--strand-names', nargs='+',
                       help='Names for strands (optional)')
    parser.add_argument('--concentrations', nargs='+', type=float,
                       help='Initial strand concentrations (M)')
    parser.add_argument('--material', choices=['DNA', 'RNA'], default='DNA',
                       help='Material type (DNA or RNA)')
    parser.add_argument('--temperature', type=float, default=37.0,
                       help='Temperature in Celsius')
    parser.add_argument('--thermal-scan', action='store_true',
                       help='Perform thermal stability analysis')
    parser.add_argument('--titration', action='store_true',
                       help='Perform titration analysis')
    parser.add_argument('--output', type=str,
                       help='Output file for results')

    args = parser.parse_args()

    # Prepare strands
    strand_names = args.strand_names or [f"strand_{i+1}" for i in range(len(args.strands))]
    strands = [MockStrand(seq, name) for seq, name in zip(args.strands, strand_names)]

    # Prepare concentrations
    concentrations = args.concentrations or [1e-6] * len(strands)
    if len(concentrations) != len(strands):
        print("Warning: Number of concentrations doesn't match number of strands")
        concentrations = [1e-6] * len(strands)

    # Initialize model
    model = MockModel(material=args.material, temperature=args.temperature)
    print(f"Initialized {args.material} model at {args.temperature}°C")

    print("\n" + "="*60)
    print("NUPACK COMPLEX EQUILIBRIUM ANALYSIS")
    print("="*60)

    # Create main tube
    tube = MockTube(strands, concentrations)

    print(f"\n1. STRAND SPECIFICATIONS")
    print("-" * 30)
    for i, (strand, conc) in enumerate(zip(strands, concentrations)):
        print(f"  {strand} at {conc:.2e} M")

    print(f"\n2. COMPLEX ENUMERATION")
    print("-" * 30)
    print(f"Found {len(tube.complexes)} possible complexes:")

    complex_data = []
    for i, complex in enumerate(tube.complexes):
        strand_names = '+'.join(s.name for s in complex.strands)
        print(f"  Complex {i+1}: {strand_names}")
        print(f"    Structure: {complex.structure}")
        print(f"    Free Energy: {complex.free_energy:.2f} kcal/mol")

        complex_data.append({
            'complex_id': i+1,
            'strands': strand_names,
            'structure': complex.structure,
            'free_energy': complex.free_energy
        })

    print(f"\n3. EQUILIBRIUM CONCENTRATIONS")
    print("-" * 30)
    fractions = tube.get_complex_fractions()

    equilibrium_data = []
    for i, (complex, conc, fraction) in enumerate(zip(tube.complexes, tube.equilibrium_concentrations, fractions)):
        strand_names = '+'.join(s.name for s in complex.strands)
        print(f"  {strand_names}: {conc:.2e} M ({fraction:.1%})")

        equilibrium_data.append({
            'complex_id': i+1,
            'concentration': conc,
            'fraction': fraction
        })

    # Find dominant complex
    max_fraction_idx = np.argmax(fractions)
    dominant_complex = tube.complexes[max_fraction_idx]
    print(f"\nDominant complex: {'+'.join(s.name for s in dominant_complex.strands)} ({fractions[max_fraction_idx]:.1%})")

    all_results = {
        'analysis_type': 'complex_equilibrium',
        'model_parameters': {
            'material': args.material,
            'temperature': args.temperature
        },
        'strands': [{'name': s.name, 'sequence': s.sequence} for s in strands],
        'initial_concentrations': concentrations,
        'complexes': complex_data,
        'equilibrium': equilibrium_data
    }

    # Thermal stability analysis
    if args.thermal_scan:
        print(f"\n4. THERMAL STABILITY ANALYSIS")
        print("-" * 30)

        thermal_data = thermal_stability_analysis(tube)

        # Show melting behavior
        print("Temperature-dependent complex fractions:")
        pivot_data = thermal_data.pivot(index='temperature', columns='complex_name', values='fraction')
        print(pivot_data.head())

        all_results['thermal_analysis'] = thermal_data.to_dict('records')

    # Titration analysis
    if args.titration and len(strands) >= 2:
        print(f"\n5. TITRATION ANALYSIS")
        print("-" * 30)

        # Use first two strands for titration
        conc_range = np.logspace(-8, -4, 20)  # 1 nM to 100 μM
        titration_data = titration_analysis(strands[0], strands[1], conc_range)

        print("Titration results (first 5 points):")
        print(titration_data.head())

        # Find apparent Kd (concentration where 50% bound)
        mid_point_idx = np.argmin(np.abs(titration_data['heterodimer_fraction'] - 0.5))
        apparent_kd = titration_data.iloc[mid_point_idx]['strand_b_concentration']
        print(f"\nApparent Kd ≈ {apparent_kd:.2e} M")

        all_results['titration_analysis'] = {
            'data': titration_data.to_dict('records'),
            'apparent_kd': apparent_kd
        }

    # Competition analysis with different ratios
    if len(strands) >= 2:
        print(f"\n6. STOICHIOMETRY ANALYSIS")
        print("-" * 30)

        ratios = [0.1, 0.5, 1.0, 2.0, 10.0]
        competition_results = []

        for ratio in ratios:
            concs = [concentrations[0], concentrations[0] * ratio]
            test_tube = MockTube(strands[:2], concs)
            test_fractions = test_tube.get_complex_fractions()

            # Find major complexes
            major_complexes = []
            for i, (complex, frac) in enumerate(zip(test_tube.complexes, test_fractions)):
                if frac > 0.1:  # Only complexes > 10%
                    major_complexes.append({
                        'strands': '+'.join(s.name for s in complex.strands),
                        'fraction': frac
                    })

            competition_results.append({
                'ratio': ratio,
                'concentrations': concs,
                'major_complexes': major_complexes
            })

            print(f"Ratio {strands[1].name}:{strands[0].name} = {ratio}:")
            for comp in major_complexes:
                print(f"  {comp['strands']}: {comp['fraction']:.1%}")

        all_results['stoichiometry_analysis'] = competition_results

    # Save results if output file specified
    if args.output:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            else:
                return convert_numpy(data)

        with open(args.output, 'w') as f:
            json.dump(clean_for_json(all_results), f, indent=2)

        print(f"\nResults saved to: {args.output}")

    print("\n" + "="*60)
    print("Complex equilibrium analysis completed!")
    print("Note: This is a mock implementation for MCP development.")
    print("For actual NUPACK functionality, install the compiled NUPACK library.")

if __name__ == '__main__':
    main()