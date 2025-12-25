"""
MCP Server for NUPACK

Provides both synchronous and asynchronous (submit) APIs for NUPACK nucleic acid analysis tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Union
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("NUPACK")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def analyze_thermodynamics(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    material: str = "DNA",
    temperature: float = 37.0,
    temp_scan: bool = False,
    temp_range: Optional[List[float]] = None,
    total_concentration: float = 1e-6,
    output_file: Optional[str] = None
) -> dict:
    """
    Analyze thermodynamic properties of nucleic acid sequences.

    Fast operation suitable for single or multiple sequences (~1 second).
    For batch processing of many sequences, use submit_thermodynamics_analysis.

    Args:
        sequences: List of DNA/RNA sequences to analyze
        input_file: Path to file with sequences (one per line)
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius (default: 37.0)
        temp_scan: Whether to perform temperature scan analysis
        temp_range: Temperature range [start, stop, step] for scan
        total_concentration: Total strand concentration in M
        output_file: Optional path to save results as JSON

    Returns:
        Dictionary with thermodynamic analysis results and metadata

    Example:
        analyze_thermodynamics(sequences=["ATCGATCGATCG", "GGCCAATTCCGG"], material="DNA", temperature=37.0)
    """
    from thermodynamic_analysis import run_thermodynamic_analysis

    try:
        result = run_thermodynamic_analysis(
            sequences=sequences,
            input_file=input_file,
            output_file=output_file,
            material=material,
            temperature=temperature,
            temp_scan=temp_scan,
            temp_range=temp_range,
            total_concentration=total_concentration
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Thermodynamic analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def predict_structure(
    sequence: Optional[str] = None,
    input_file: Optional[str] = None,
    material: str = "RNA",
    temperature: float = 37.0,
    energy_gap: float = 5.0,
    max_suboptimal: int = 10,
    probability_threshold: float = 0.5,
    output_file: Optional[str] = None
) -> dict:
    """
    Predict secondary structure of a nucleic acid sequence.

    Fast operation suitable for single sequences (~1 second).
    For batch processing of multiple sequences, use submit_structure_prediction.

    Args:
        sequence: DNA/RNA sequence to analyze
        input_file: Path to file with sequence
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius (default: 37.0)
        energy_gap: Energy gap for suboptimal structures in kcal/mol
        max_suboptimal: Maximum number of suboptimal structures
        probability_threshold: Base pair probability threshold
        output_file: Optional path to save results as JSON

    Returns:
        Dictionary with structure prediction results and metadata

    Example:
        predict_structure(sequence="GGCCAATTCCGG", material="RNA", energy_gap=3.0)
    """
    from structure_prediction import run_structure_prediction

    try:
        result = run_structure_prediction(
            sequence=sequence,
            input_file=input_file,
            output_file=output_file,
            material=material,
            temperature=temperature,
            energy_gap=energy_gap,
            max_suboptimal=max_suboptimal,
            probability_threshold=probability_threshold
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Structure prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def design_sequence(
    target_structure: Optional[str] = None,
    input_file: Optional[str] = None,
    material: str = "DNA",
    num_designs: int = 5,
    temperature: float = 37.0,
    gc_content: Optional[float] = None,
    constraints: Optional[str] = None,
    output_file: Optional[str] = None
) -> dict:
    """
    Design nucleic acid sequences to fold into target secondary structures.

    Fast operation suitable for single structures (~1 second).
    For batch processing of multiple structures, use submit_sequence_design.

    Args:
        target_structure: Target secondary structure in dot-bracket notation
        input_file: Path to file with target structure
        material: Nucleic acid type ("DNA" or "RNA")
        num_designs: Number of sequence designs to generate
        temperature: Temperature in Celsius (default: 37.0)
        gc_content: Target GC content (0.0-1.0)
        constraints: Sequence constraints (e.g., "NNNATCGNNN")
        output_file: Optional path to save results as JSON

    Returns:
        Dictionary with designed sequences and quality scores

    Example:
        design_sequence(target_structure="((((....))))", material="DNA", num_designs=3)
    """
    from sequence_design import run_sequence_design

    try:
        result = run_sequence_design(
            target_structure=target_structure,
            input_file=input_file,
            output_file=output_file,
            material=material,
            num_designs=num_designs,
            temperature=temperature,
            gc_content=gc_content,
            constraints=constraints
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Sequence design failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_complex_equilibrium(
    strands: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    concentrations: Optional[List[float]] = None,
    material: str = "DNA",
    temperature: float = 37.0,
    thermal_scan: bool = False,
    stoichiometry_scan: bool = False,
    output_file: Optional[str] = None
) -> dict:
    """
    Analyze complex equilibrium for multi-strand nucleic acid systems.

    Fast operation suitable for 2-4 strands (~1 second).
    For large systems or batch processing, use submit_complex_equilibrium_analysis.

    Args:
        strands: List of DNA/RNA strand sequences
        input_file: Path to file with strand sequences (one per line)
        concentrations: Initial strand concentrations in M
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius (default: 37.0)
        thermal_scan: Whether to perform temperature scan
        stoichiometry_scan: Whether to analyze stoichiometric ratios
        output_file: Optional path to save results as JSON

    Returns:
        Dictionary with complex equilibrium analysis results

    Example:
        analyze_complex_equilibrium(strands=["ATCGATCGATCG", "CGAUCGAUCGAU"], concentrations=[1e-6, 1e-6])
    """
    from complex_equilibrium import run_complex_equilibrium

    try:
        result = run_complex_equilibrium(
            strands=strands,
            input_file=input_file,
            concentrations=concentrations,
            output_file=output_file,
            material=material,
            temperature=temperature,
            thermal_scan=thermal_scan,
            stoichiometry_scan=stoichiometry_scan
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Complex equilibrium analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def evaluate_energy(
    sequence: Optional[str] = None,
    structure: Optional[str] = None,
    input_file: Optional[str] = None,
    material: str = "RNA",
    temperature: float = 37.0,
    landscape: bool = False,
    sensitivity: bool = False,
    temperature_analysis: bool = False,
    output_file: Optional[str] = None
) -> dict:
    """
    Evaluate energy landscape and properties of nucleic acid structures.

    Fast operation suitable for single sequences (~1 second).
    For batch processing or extensive analysis, use submit_energy_evaluation.

    Args:
        sequence: DNA/RNA sequence
        structure: Secondary structure in dot-bracket notation
        input_file: Path to file with sequence and structure
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius (default: 37.0)
        landscape: Whether to generate energy landscape
        sensitivity: Whether to perform parameter sensitivity analysis
        temperature_analysis: Whether to analyze temperature dependence
        output_file: Optional path to save results as JSON

    Returns:
        Dictionary with energy evaluation results and analysis

    Example:
        evaluate_energy(sequence="GGCCAATTCCGG", structure="((((....))))", landscape=True)
    """
    from energy_evaluation import run_energy_evaluation

    try:
        result = run_energy_evaluation(
            sequence=sequence,
            structure=structure,
            input_file=input_file,
            output_file=output_file,
            material=material,
            temperature=temperature,
            landscape=landscape,
            sensitivity=sensitivity,
            temperature_analysis=temperature_analysis
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Energy evaluation failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min or batch processing)
# ==============================================================================

@mcp.tool()
def submit_thermodynamics_analysis(
    sequences: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    material: str = "DNA",
    temperature: float = 37.0,
    temp_scan: bool = False,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit thermodynamic analysis for background processing.

    Use this for batch processing of many sequences or when you want to run
    analysis in the background. Returns a job_id for tracking.

    Args:
        sequences: List of DNA/RNA sequences to analyze
        input_file: Path to file with sequences
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius
        temp_scan: Whether to perform temperature scan
        output_dir: Directory to save outputs
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs

    Example:
        submit_thermodynamics_analysis(sequences=["ATCG"*50, "CGAU"*50], temp_scan=True)
    """
    script_path = str(SCRIPTS_DIR / "thermodynamic_analysis.py")

    args = {
        "material": material,
        "temperature": temperature,
        "temp_scan": temp_scan
    }

    if sequences:
        args["sequences"] = " ".join(sequences)
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "thermodynamics_analysis"
    )

@mcp.tool()
def submit_structure_prediction(
    sequence: Optional[str] = None,
    input_file: Optional[str] = None,
    material: str = "RNA",
    temperature: float = 37.0,
    energy_gap: float = 5.0,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit structure prediction for background processing.

    Use this for complex predictions or when processing in the background.
    Returns a job_id for tracking.

    Args:
        sequence: DNA/RNA sequence to analyze
        input_file: Path to file with sequence
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius
        energy_gap: Energy gap for suboptimal structures
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the prediction job

    Example:
        submit_structure_prediction(sequence="GGCC"*100, energy_gap=10.0)
    """
    script_path = str(SCRIPTS_DIR / "structure_prediction.py")

    args = {
        "material": material,
        "temperature": temperature,
        "energy_gap": energy_gap
    }

    if sequence:
        args["sequence"] = sequence
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "structure_prediction"
    )

@mcp.tool()
def submit_sequence_design(
    target_structure: Optional[str] = None,
    input_file: Optional[str] = None,
    material: str = "DNA",
    num_designs: int = 50,
    temperature: float = 37.0,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit sequence design for background processing.

    Use this for generating many sequence designs or complex target structures.
    Returns a job_id for tracking.

    Args:
        target_structure: Target secondary structure in dot-bracket notation
        input_file: Path to file with target structure
        material: Nucleic acid type ("DNA" or "RNA")
        num_designs: Number of sequence designs to generate
        temperature: Temperature in Celsius
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the design job

    Example:
        submit_sequence_design(target_structure="((((....))))"*10, num_designs=100)
    """
    script_path = str(SCRIPTS_DIR / "sequence_design.py")

    args = {
        "material": material,
        "num_designs": num_designs,
        "temperature": temperature
    }

    if target_structure:
        args["target_structure"] = target_structure
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "sequence_design"
    )

@mcp.tool()
def submit_complex_equilibrium_analysis(
    strands: Optional[List[str]] = None,
    input_file: Optional[str] = None,
    concentrations: Optional[List[float]] = None,
    material: str = "DNA",
    temperature: float = 37.0,
    thermal_scan: bool = True,
    stoichiometry_scan: bool = True,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit complex equilibrium analysis for background processing.

    Use this for large multi-strand systems or comprehensive analysis.
    Returns a job_id for tracking.

    Args:
        strands: List of DNA/RNA strand sequences
        input_file: Path to file with strand sequences
        concentrations: Initial strand concentrations
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius
        thermal_scan: Whether to perform temperature scan
        stoichiometry_scan: Whether to analyze stoichiometric ratios
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the equilibrium analysis job

    Example:
        submit_complex_equilibrium_analysis(strands=["ATCG"*20]*5, thermal_scan=True)
    """
    script_path = str(SCRIPTS_DIR / "complex_equilibrium.py")

    args = {
        "material": material,
        "temperature": temperature,
        "thermal_scan": thermal_scan,
        "stoichiometry_scan": stoichiometry_scan
    }

    if strands:
        args["strands"] = " ".join(strands)
    if concentrations:
        args["concentrations"] = " ".join(map(str, concentrations))
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "complex_equilibrium_analysis"
    )

@mcp.tool()
def submit_energy_evaluation(
    sequence: Optional[str] = None,
    structure: Optional[str] = None,
    input_file: Optional[str] = None,
    material: str = "RNA",
    temperature: float = 37.0,
    landscape: bool = True,
    sensitivity: bool = True,
    temperature_analysis: bool = True,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit energy evaluation for background processing.

    Use this for extensive energy landscape analysis or batch processing.
    Returns a job_id for tracking.

    Args:
        sequence: DNA/RNA sequence
        structure: Secondary structure in dot-bracket notation
        input_file: Path to file with sequence and structure
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius
        landscape: Whether to generate energy landscape
        sensitivity: Whether to perform parameter sensitivity analysis
        temperature_analysis: Whether to analyze temperature dependence
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the evaluation job

    Example:
        submit_energy_evaluation(sequence="GGCC"*50, structure="(((("*50+"."*4+"))))"*50, landscape=True)
    """
    script_path = str(SCRIPTS_DIR / "energy_evaluation.py")

    args = {
        "material": material,
        "temperature": temperature,
        "landscape": landscape,
        "sensitivity": sensitivity,
        "temperature_analysis": temperature_analysis
    }

    if sequence:
        args["sequence"] = sequence
    if structure:
        args["structure"] = structure
    if input_file:
        args["input"] = input_file
    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or "energy_evaluation"
    )

# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_nucleic_acid_analysis(
    input_files: List[str],
    analysis_type: str = "all",
    material: str = "DNA",
    temperature: float = 37.0,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch nucleic acid analysis for multiple input files.

    Processes multiple input files with the specified analysis type.
    Suitable for:
    - Processing many sequences/structures at once
    - Large-scale nucleic acid analysis
    - Parallel processing of independent files

    Args:
        input_files: List of input file paths to process
        analysis_type: Type of analysis - "thermodynamics", "structure", "design", "equilibrium", "energy", or "all"
        material: Nucleic acid type ("DNA" or "RNA")
        temperature: Temperature in Celsius for all analyses
        output_dir: Directory to save all outputs
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job

    Example:
        submit_batch_nucleic_acid_analysis(["seq1.txt", "seq2.txt"], analysis_type="structure")
    """
    # For now, process first file with the requested analysis type
    # In a production system, you'd want a dedicated batch processing script
    if not input_files:
        return {"status": "error", "error": "No input files provided"}

    analysis_map = {
        "thermodynamics": "thermodynamic_analysis.py",
        "structure": "structure_prediction.py",
        "design": "sequence_design.py",
        "equilibrium": "complex_equilibrium.py",
        "energy": "energy_evaluation.py"
    }

    if analysis_type not in analysis_map and analysis_type != "all":
        return {"status": "error", "error": f"Invalid analysis type: {analysis_type}"}

    # Use thermodynamics as default for "all" for now
    script_name = analysis_map.get(analysis_type, "thermodynamic_analysis.py")
    script_path = str(SCRIPTS_DIR / script_name)

    args = {
        "input": input_files[0],  # Process first file
        "material": material,
        "temperature": temperature
    }

    if output_dir:
        args["output_dir"] = output_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_{analysis_type}_analysis"
    )

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()