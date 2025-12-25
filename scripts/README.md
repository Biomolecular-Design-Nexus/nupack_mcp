# NUPACK MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported (`numpy`, `pandas`, `json`, `argparse`, `pathlib`)
2. **Self-Contained**: Functions inlined where possible, minimal repo dependencies
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts Overview

| Script | Description | Dependencies | Config File |
|--------|-------------|--------------|-------------|
| `thermodynamic_analysis.py` | Analyze thermodynamic properties of nucleic acids | Standard only | `configs/thermodynamic_analysis_config.json` |
| `structure_prediction.py` | Predict secondary structure from sequence | Standard only | `configs/structure_prediction_config.json` |
| `sequence_design.py` | Design sequences to fold into target structures | Standard + random | `configs/sequence_design_config.json` |
| `complex_equilibrium.py` | Analyze multi-strand complex equilibrium | Standard + random, optional matplotlib | `configs/complex_equilibrium_config.json` |
| `energy_evaluation.py` | Evaluate energy landscape and structure properties | Standard only | `configs/energy_evaluation_config.json` |

## Dependencies Summary

### Essential (all scripts)
- `numpy`: Numerical calculations
- `pandas`: Data handling and analysis
- `json`: Configuration and output formatting
- `argparse`: Command line interface
- `pathlib`: File path handling
- `typing`: Type hints

### Additional
- `random`: Used in sequence_design.py and complex_equilibrium.py
- `matplotlib`: Optional in complex_equilibrium.py (graceful fallback)

### Repo Dependencies
**None** - All scripts are completely independent of the repo directory.

## Usage

### Basic Usage

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Run individual scripts
python scripts/thermodynamic_analysis.py --sequences ATCGATCGATCG GGCCAATTCCGG --output results/thermo.json
python scripts/structure_prediction.py --sequence GGCCAATTCCGG --output results/structure.json
python scripts/sequence_design.py --target-structure "((((....))))" --num-designs 5 --output results/design.json
python scripts/complex_equilibrium.py --strands ATCG CGAU --concentrations 1e-6 1e-6 --output results/equilibrium.json
python scripts/energy_evaluation.py --sequence GGCC --structure "((..))" --output results/energy.json
```

### Using Configuration Files

```bash
# Use predefined configurations
python scripts/thermodynamic_analysis.py --config configs/thermodynamic_analysis_config.json --sequences ATCG CGAU
python scripts/structure_prediction.py --config configs/structure_prediction_config.json --sequence GGCCAATTCCGG

# Override config parameters
python scripts/sequence_design.py --config configs/sequence_design_config.json --num-designs 10 --material RNA
```

### Input/Output Formats

#### Input Formats
- **Sequences**: Command line arguments or files with one sequence per line
- **Structures**: Dot-bracket notation (e.g., `"(((...)))"`)
- **Configuration**: JSON files

#### Output Format
All scripts output JSON with consistent structure:
```json
{
  "analysis_type": "thermodynamic|structure|design|equilibrium|energy",
  "model_parameters": {...},
  "results": {...},
  "metadata": {...}
}
```

## Shared Library

Common functions are in `scripts/lib/`:

### `lib/io.py` - File I/O utilities
- `load_json()`, `save_json()`: JSON file handling
- `load_sequences()`, `save_sequences()`: Sequence file handling
- `ensure_output_dir()`: Directory creation

### `lib/structure_utils.py` - Structure analysis
- `extract_base_pairs()`: Parse dot-bracket notation
- `validate_structure()`: Structure validation
- `get_gc_content()`: Calculate GC content
- `complement_base()`, `can_pair()`: Base pairing utilities

### `lib/config_utils.py` - Configuration management
- `load_config()`: Load JSON configuration files
- `merge_configs()`: Merge multiple config sources
- `validate_config_keys()`: Configuration validation

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped as MCP tools:

```python
# Example MCP tool wrapper
from scripts.thermodynamic_analysis import run_thermodynamic_analysis

@mcp.tool()
def analyze_thermodynamics(sequences: List[str], material: str = "DNA", temperature: float = 37.0):
    """Analyze thermodynamic properties of nucleic acid sequences."""
    return run_thermodynamic_analysis(sequences=sequences, material=material, temperature=temperature)
```

### MCP Function Signatures

All scripts follow this pattern:
```python
def run_<script_name>(
    # Input data
    input_data: Union[str, List[str], Path] = None,
    input_file: Optional[Union[str, Path]] = None,

    # Output
    output_file: Optional[Union[str, Path]] = None,

    # Configuration
    config: Optional[Dict[str, Any]] = None,

    # Overrides
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for <script purpose>.
    Returns dict with results, output_file path, and metadata.
    """
```

## Configuration

### Configuration File Structure
```json
{
  "_description": "Human-readable description",
  "_source": "Original use case script",

  "model": {
    "material": "DNA|RNA",
    "temperature": 37.0
  },

  "analysis": {
    // Analysis-specific parameters
  },

  "output": {
    "format": "json",
    "include_metadata": true
  },

  "defaults": {
    // Default input values for testing
  }
}
```

### Configuration Priority (highest to lowest)
1. Command line arguments (`--temperature 25`)
2. User config file (`--config my_config.json`)
3. Script defaults

## Examples

### Thermodynamic Analysis
```bash
# Basic thermodynamic analysis
python scripts/thermodynamic_analysis.py \
  --sequences ATCGATCGATCG GGCCAATTCCGG \
  --material DNA \
  --temperature 37 \
  --output results/thermo.json

# With temperature scanning
python scripts/thermodynamic_analysis.py \
  --sequences ATCGATCGATCG \
  --temp-scan \
  --output results/temp_scan.json
```

### Structure Prediction
```bash
# Predict RNA secondary structure
python scripts/structure_prediction.py \
  --sequence GGCCAAUUCCGG \
  --material RNA \
  --energy-gap 3.0 \
  --output results/structure.json
```

### Sequence Design
```bash
# Design sequences for target structure
python scripts/sequence_design.py \
  --target-structure "((((....))))" \
  --material DNA \
  --num-designs 10 \
  --output results/designs.json
```

### Complex Equilibrium
```bash
# Analyze two-strand complex formation
python scripts/complex_equilibrium.py \
  --strands ATCGATCGATCG CGAUCGAUCGAU \
  --concentrations 1e-6 1e-6 \
  --thermal-scan \
  --output results/equilibrium.json
```

### Energy Evaluation
```bash
# Evaluate structure energy and landscape
python scripts/energy_evaluation.py \
  --sequence GGCCAATTCCGG \
  --structure "((((....))))" \
  --landscape \
  --sensitivity \
  --output results/energy.json
```

## Testing

```bash
# Test all scripts with example data
./scripts/test_all.sh

# Test individual scripts
python scripts/thermodynamic_analysis.py --sequences ATCG --output /tmp/test.json
python scripts/structure_prediction.py --sequence GGCC --output /tmp/test.json
```

## Notes

- All scripts are **mock implementations** designed for MCP server development
- Real NUPACK functionality would require the compiled NUPACK library
- Parameter files are authentic NUPACK thermodynamic data
- Execution times are excellent for development/testing workflows (<1 second)
- Scripts provide clear documentation and error messages
- **Ready for MCP server integration** without additional modifications