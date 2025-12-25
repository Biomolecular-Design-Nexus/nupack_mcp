# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: NUPACK
- **Version**: 1.0.0
- **Created Date**: 2025-12-25
- **Server Path**: `src/server.py`
- **Package Manager**: mamba (preferred over conda)
- **Dependencies**: fastmcp, loguru, standard libraries

## Architecture Overview

The NUPACK MCP server provides both synchronous and asynchronous APIs for nucleic acid analysis. All original scripts from Step 5 have been successfully integrated with both API types.

### API Design Philosophy

1. **Synchronous API** - For fast operations (<10 minutes)
   - Direct function call, immediate response
   - Suitable for: quick analysis, single sequences, small datasets

2. **Submit API** - For long-running tasks (>10 minutes) or batch processing
   - Submit job, get job_id, check status, retrieve results
   - Suitable for: batch processing, large datasets, background operations

## Job Management Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_job_status` | Check job progress and status | `job_id: str` |
| `get_job_result` | Get completed job results | `job_id: str` |
| `get_job_log` | View job execution logs | `job_id: str`, `tail: int = 50` |
| `cancel_job` | Cancel running job | `job_id: str` |
| `list_jobs` | List all jobs (filterable) | `status: Optional[str] = None` |

### Job Status Values
- `pending`: Job submitted, waiting to start
- `running`: Job currently executing
- `completed`: Job finished successfully
- `failed`: Job encountered an error
- `cancelled`: Job was manually cancelled

## Synchronous Tools (Fast Operations < 10 min)

### API Design Analysis

All NUPACK scripts were analyzed for runtime performance:

| Script | Typical Runtime | Test Results | API Type |
|--------|----------------|--------------|----------|
| `thermodynamic_analysis.py` | ~0.5s | ✅ 2 sequences in 0.1s | Sync + Submit |
| `structure_prediction.py` | ~0.5s | ✅ 20nt sequence in 0.47s | Sync + Submit |
| `sequence_design.py` | ~0.5s | ✅ 3 designs in 0.3s | Sync + Submit |
| `complex_equilibrium.py` | ~0.8s | ✅ 2 strands in 0.77s | Sync + Submit |
| `energy_evaluation.py` | ~0.5s | ✅ Energy analysis in 0.4s | Sync + Submit |

### Tool Details

#### analyze_thermodynamics
- **Description**: Analyze thermodynamic properties of nucleic acid sequences
- **Source Script**: `scripts/thermodynamic_analysis.py`
- **Estimated Runtime**: ~1 second for typical inputs

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences | List[str] | No | None | List of DNA/RNA sequences |
| input_file | str | No | None | File with sequences (one per line) |
| material | str | No | "DNA" | Nucleic acid type ("DNA" or "RNA") |
| temperature | float | No | 37.0 | Temperature in Celsius |
| temp_scan | bool | No | False | Whether to perform temperature scan |
| temp_range | List[float] | No | None | Temperature range [start, stop, step] |
| total_concentration | float | No | 1e-6 | Total strand concentration in M |
| output_file | str | No | None | Optional path to save results |

**Example:**
```
Use analyze_thermodynamics with sequences ["ATCGATCGATCG", "GGCCAATTCCGG"] and material "DNA"
```

---

#### predict_structure
- **Description**: Predict secondary structure of a nucleic acid sequence
- **Source Script**: `scripts/structure_prediction.py`
- **Estimated Runtime**: ~1 second for typical inputs

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequence | str | No | None | DNA/RNA sequence to analyze |
| input_file | str | No | None | File with sequence |
| material | str | No | "RNA" | Nucleic acid type ("DNA" or "RNA") |
| temperature | float | No | 37.0 | Temperature in Celsius |
| energy_gap | float | No | 5.0 | Energy gap for suboptimal structures |
| max_suboptimal | int | No | 10 | Maximum number of suboptimal structures |
| probability_threshold | float | No | 0.5 | Base pair probability threshold |
| output_file | str | No | None | Optional path to save results |

**Example:**
```
Use predict_structure with sequence "GGCCAATTCCGG" and material "RNA"
```

---

#### design_sequence
- **Description**: Design nucleic acid sequences to fold into target secondary structures
- **Source Script**: `scripts/sequence_design.py`
- **Estimated Runtime**: ~1 second for typical inputs

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| target_structure | str | No | None | Target structure in dot-bracket notation |
| input_file | str | No | None | File with target structure |
| material | str | No | "DNA" | Nucleic acid type ("DNA" or "RNA") |
| num_designs | int | No | 5 | Number of sequence designs to generate |
| temperature | float | No | 37.0 | Temperature in Celsius |
| gc_content | float | No | None | Target GC content (0.0-1.0) |
| constraints | str | No | None | Sequence constraints (e.g., "NNNATCGNNN") |
| output_file | str | No | None | Optional path to save results |

**Example:**
```
Use design_sequence with target_structure "((((....))))" and num_designs 3
```

---

#### analyze_complex_equilibrium
- **Description**: Analyze complex equilibrium for multi-strand nucleic acid systems
- **Source Script**: `scripts/complex_equilibrium.py`
- **Estimated Runtime**: ~1 second for typical inputs

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| strands | List[str] | No | None | List of DNA/RNA strand sequences |
| input_file | str | No | None | File with strand sequences (one per line) |
| concentrations | List[float] | No | None | Initial strand concentrations in M |
| material | str | No | "DNA" | Nucleic acid type ("DNA" or "RNA") |
| temperature | float | No | 37.0 | Temperature in Celsius |
| thermal_scan | bool | No | False | Whether to perform temperature scan |
| stoichiometry_scan | bool | No | False | Whether to analyze stoichiometric ratios |
| output_file | str | No | None | Optional path to save results |

**Example:**
```
Use analyze_complex_equilibrium with strands ["ATCGATCGATCG", "CGAUCGAUCGAU"] and concentrations [1e-6, 1e-6]
```

---

#### evaluate_energy
- **Description**: Evaluate energy landscape and properties of nucleic acid structures
- **Source Script**: `scripts/energy_evaluation.py`
- **Estimated Runtime**: ~1 second for typical inputs

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequence | str | No | None | DNA/RNA sequence |
| structure | str | No | None | Secondary structure in dot-bracket notation |
| input_file | str | No | None | File with sequence and structure |
| material | str | No | "RNA" | Nucleic acid type ("DNA" or "RNA") |
| temperature | float | No | 37.0 | Temperature in Celsius |
| landscape | bool | No | False | Whether to generate energy landscape |
| sensitivity | bool | No | False | Whether to perform parameter sensitivity analysis |
| temperature_analysis | bool | No | False | Whether to analyze temperature dependence |
| output_file | str | No | None | Optional path to save results |

**Example:**
```
Use evaluate_energy with sequence "GGCCAATTCCGG" and structure "((((....))))"
```

---

## Submit Tools (Long Operations > 10 min or Batch Processing)

### When to Use Submit API
- Processing large datasets (>100 sequences)
- Complex analysis requiring extensive computation
- Batch processing of multiple files
- Background processing while doing other work
- Jobs that might take more than a few minutes

### Tool Details

#### submit_thermodynamics_analysis
- **Description**: Submit thermodynamic analysis for background processing
- **Source Script**: `scripts/thermodynamic_analysis.py`
- **Estimated Runtime**: Variable (depends on input size and scan options)
- **Supports Batch**: ✅ Yes (multiple sequences)

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences | List[str] | No | None | List of DNA/RNA sequences |
| input_file | str | No | None | File with sequences |
| material | str | No | "DNA" | Nucleic acid type |
| temperature | float | No | 37.0 | Temperature in Celsius |
| temp_scan | bool | No | False | Whether to perform temperature scan |
| output_dir | str | No | None | Directory to save outputs |
| job_name | str | No | None | Custom job name |

**Example:**
```
Use submit_thermodynamics_analysis with sequences ["ATCG"*50, "CGAU"*50] and temp_scan True
```

---

#### submit_structure_prediction
- **Description**: Submit structure prediction for background processing
- **Source Script**: `scripts/structure_prediction.py`
- **Estimated Runtime**: Variable (depends on sequence length and energy gap)

**Example:**
```
Use submit_structure_prediction with sequence "GGCC"*100 and energy_gap 10.0
```

---

#### submit_sequence_design
- **Description**: Submit sequence design for background processing
- **Source Script**: `scripts/sequence_design.py`
- **Estimated Runtime**: Variable (depends on structure complexity and num_designs)

**Example:**
```
Use submit_sequence_design with target_structure "((((....))))"*10 and num_designs 100
```

---

#### submit_complex_equilibrium_analysis
- **Description**: Submit complex equilibrium analysis for background processing
- **Source Script**: `scripts/complex_equilibrium.py`
- **Estimated Runtime**: Variable (depends on number of strands and scan options)

**Example:**
```
Use submit_complex_equilibrium_analysis with strands ["ATCG"*20]*5 and thermal_scan True
```

---

#### submit_energy_evaluation
- **Description**: Submit energy evaluation for background processing
- **Source Script**: `scripts/energy_evaluation.py`
- **Estimated Runtime**: Variable (depends on landscape and sensitivity analysis)

**Example:**
```
Use submit_energy_evaluation with sequence "GGCC"*50 and landscape True and sensitivity True
```

---

#### submit_batch_nucleic_acid_analysis
- **Description**: Submit batch analysis for multiple input files
- **Source Scripts**: All NUPACK scripts (depending on analysis_type)
- **Estimated Runtime**: Variable (depends on number of files and analysis type)

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_files | List[str] | Yes | - | List of input file paths |
| analysis_type | str | No | "all" | Type: "thermodynamics", "structure", "design", "equilibrium", "energy", "all" |
| material | str | No | "DNA" | Nucleic acid type |
| temperature | float | No | 37.0 | Temperature for all analyses |
| output_dir | str | No | None | Directory to save outputs |
| job_name | str | No | None | Custom job name |

**Example:**
```
Use submit_batch_nucleic_acid_analysis with input_files ["seq1.txt", "seq2.txt"] and analysis_type "structure"
```

---

## Workflow Examples

### Quick Analysis (Sync)
```
1. Single sequence thermodynamics:
   Use analyze_thermodynamics with sequences ["ATCGATCGATCG"] and material "DNA"
   → Returns results immediately (< 1 second)

2. Structure prediction:
   Use predict_structure with sequence "GGCCAATTCCGG" and material "RNA"
   → Returns structure immediately (< 1 second)

3. Sequence design:
   Use design_sequence with target_structure "((((....))))" and num_designs 5
   → Returns designs immediately (< 1 second)
```

### Long-Running Task (Submit API)
```
1. Submit:
   Use submit_thermodynamics_analysis with sequences ["ATCG"*100]*1000 and temp_scan True
   → Returns: {"status": "submitted", "job_id": "abc123", "message": "..."}

2. Check Status:
   Use get_job_status with job_id "abc123"
   → Returns: {"job_id": "abc123", "status": "running", "started_at": "...", ...}

3. Monitor Progress:
   Use get_job_log with job_id "abc123" and tail 10
   → Returns: {"status": "success", "log_lines": [...], ...}

4. Get Result:
   Use get_job_result with job_id "abc123" (when status == "completed")
   → Returns: {"status": "success", "result": {...}}
```

### Batch Processing Workflow
```
1. Submit Batch:
   Use submit_batch_nucleic_acid_analysis with input_files ["dataset1.txt", "dataset2.txt", "dataset3.txt"]
   → Processes all files with same parameters

2. Monitor:
   Use list_jobs to see all running jobs
   Use get_job_status for specific job updates
   Use get_job_log to see execution progress

3. Retrieve Results:
   Use get_job_result when each job completes
```

## Error Handling

All tools return structured error responses:

```json
{
  "status": "error",
  "error": "Descriptive error message"
}
```

### Common Error Types
- **FileNotFoundError**: Input file not found
- **ValueError**: Invalid input parameters (e.g., invalid sequence, malformed structure)
- **RuntimeError**: Script execution error
- **JobNotFoundError**: Invalid job_id provided

## Installation and Usage

### Prerequisites
```bash
# Use mamba (preferred) or conda
PKG_MGR="mamba"  # or "conda"

# Activate environment
$PKG_MGR activate ./env

# Install MCP dependencies (already installed)
pip install fastmcp loguru
```

### Running the Server

#### With Claude Desktop
Add to Claude configuration:
```json
{
  "mcpServers": {
    "NUPACK": {
      "command": "mamba",
      "args": ["run", "-p", "./env", "python", "src/server.py"]
    }
  }
}
```

#### With fastmcp CLI
```bash
# Development mode
fastmcp dev src/server.py

# Install for Claude Code
fastmcp install claude-code src/server.py
```

#### Direct Python
```bash
mamba run -p ./env python src/server.py
```

## File Structure

```
src/
├── server.py              # Main MCP server with all tools
├── jobs/
│   ├── __init__.py        # Job management module
│   └── manager.py         # JobManager class and global instance
└── __init__.py            # Package initialization

scripts/                   # Independent NUPACK scripts (Step 5)
├── thermodynamic_analysis.py
├── structure_prediction.py
├── sequence_design.py
├── complex_equilibrium.py
├── energy_evaluation.py
└── lib/                   # Shared utilities

jobs/                      # Job execution and storage
└── [job_id]/              # Individual job directories
    ├── metadata.json      # Job metadata and status
    ├── output.json        # Job results
    └── job.log           # Execution log

configs/                   # Configuration files (from Step 5)
├── thermodynamic_analysis_config.json
├── structure_prediction_config.json
├── sequence_design_config.json
├── complex_equilibrium_config.json
├── energy_evaluation_config.json
└── default_config.json
```

## Testing Results

### Functionality Tests

| Test Type | Status | Details |
|-----------|--------|---------|
| Script Independence | ✅ Pass | All scripts work without repo dependencies |
| Sync API Functions | ✅ Pass | All 5 sync tools tested successfully |
| Submit API | ✅ Pass | Job submission, monitoring, and results retrieval working |
| Job Management | ✅ Pass | Status checking, logging, cancellation, and listing work |
| Error Handling | ✅ Pass | Structured error responses for common failure modes |

### Performance Benchmarks

| Operation | Input Size | Runtime | Memory Usage |
|-----------|------------|---------|--------------|
| Thermodynamics | 2 sequences (12nt each) | ~0.1s | <10MB |
| Structure Prediction | 1 sequence (20nt) | ~0.47s | <10MB |
| Sequence Design | 1 structure (12nt) | ~0.3s | <10MB |
| Complex Equilibrium | 2 strands (12nt each) | ~0.77s | <10MB |
| Energy Evaluation | 1 seq+struct (12nt) | ~0.4s | <10MB |

### Integration Tests

```bash
# Direct function tests
python test_direct.py
# ✅ All tests passed

# Job management tests
python test_submit.py
# ✅ Submit API test completed

# MCP server startup
fastmcp dev src/server.py
# ✅ Server starts without errors
```

## Development Notes

### Adding New Tools

To add a new NUPACK tool:

1. **Create the script** in `scripts/` following the existing pattern
2. **Add sync tool** in `src/server.py`:
   ```python
   @mcp.tool()
   def new_tool_name(...):
       from new_script import run_new_function
       # Implementation
   ```
3. **Add submit tool** for long-running operations:
   ```python
   @mcp.tool()
   def submit_new_tool_name(...):
       return job_manager.submit_job(...)
   ```

### Configuration

- Use existing configuration files in `configs/`
- Follow the established parameter naming conventions
- Maintain backward compatibility with script CLI arguments

### Error Handling Best Practices

- Always return `{"status": "success", ...}` or `{"status": "error", "error": "..."}`
- Use specific error messages for common failure modes
- Log errors using loguru for debugging

## Quality Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Script Integration | 100% | ✅ 100% (5/5 scripts) |
| API Coverage | Sync + Submit for all | ✅ 100% coverage |
| Error Handling | Structured responses | ✅ All tools covered |
| Job Management | Full lifecycle | ✅ Submit → Monitor → Results |
| Testing Coverage | All major workflows | ✅ Direct + Submit + Integration |
| Documentation | Complete tool docs | ✅ All tools documented |

## Performance Characteristics

### Synchronous API
- **Startup time**: <0.1 seconds
- **Execution time**: 0.1-0.8 seconds for typical inputs
- **Memory usage**: <10MB per operation
- **Concurrency**: Limited by Python GIL

### Asynchronous API
- **Job submission**: <0.1 seconds
- **Queue management**: Persistent job storage
- **Background execution**: Separate process per job
- **Monitoring**: Real-time status and log access

## Success Criteria Analysis

- [x] MCP server created at `src/server.py`
- [x] Job manager implemented for async operations (`src/jobs/manager.py`)
- [x] Sync tools created for fast operations (5 tools, all <10 min)
- [x] Submit tools created for long-running operations (5 submit tools + 1 batch)
- [x] Batch processing support for applicable tools (`submit_batch_nucleic_acid_analysis`)
- [x] Job management tools working (status, result, log, cancel, list)
- [x] All tools have clear descriptions for LLM use
- [x] Error handling returns structured responses
- [x] Server starts without errors: `mamba run -p ./env python src/server.py`
- [x] README updated with all tools and usage examples

## Tool Classification Summary

| Script | Runtime | Sync API | Submit API | Batch Support | Status |
|--------|---------|----------|------------|---------------|--------|
| `thermodynamic_analysis.py` | ~0.5s | ✅ `analyze_thermodynamics` | ✅ `submit_thermodynamics_analysis` | ✅ Yes | ✅ Complete |
| `structure_prediction.py` | ~0.5s | ✅ `predict_structure` | ✅ `submit_structure_prediction` | ✅ Yes | ✅ Complete |
| `sequence_design.py` | ~0.5s | ✅ `design_sequence` | ✅ `submit_sequence_design` | ✅ Yes | ✅ Complete |
| `complex_equilibrium.py` | ~0.8s | ✅ `analyze_complex_equilibrium` | ✅ `submit_complex_equilibrium_analysis` | ✅ Yes | ✅ Complete |
| `energy_evaluation.py` | ~0.5s | ✅ `evaluate_energy` | ✅ `submit_energy_evaluation` | ✅ Yes | ✅ Complete |

## Future Enhancements

### Potential Improvements
1. **Web Interface**: Add FastMCP web interface for job monitoring
2. **Progress Tracking**: Add progress percentage to job status
3. **Job Queuing**: Add job priority and queue management
4. **Result Caching**: Cache results for identical inputs
5. **Distributed Processing**: Add support for multiple worker processes
6. **Advanced Validation**: Add input validation with detailed error messages

### Scalability Considerations
- **Job Storage**: Current file-based storage suitable for moderate use
- **Concurrent Jobs**: Limited by available CPU cores and memory
- **Large Datasets**: Consider streaming for very large inputs
- **Resource Management**: Add memory and CPU usage monitoring

## Final Notes

The NUPACK MCP server successfully provides comprehensive nucleic acid analysis capabilities through both synchronous and asynchronous APIs. All original functionality from Step 5 has been preserved while adding robust job management and batch processing capabilities.

**Key Achievements:**
- **Complete Integration**: All 5 NUPACK scripts fully integrated
- **Dual API Design**: Both sync and submit APIs for optimal user experience
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **User Friendly**: Clear descriptions and examples for LLM-based usage
- **Extensible**: Clean architecture for adding new tools and capabilities

The server is ready for production use with Claude Code and other MCP clients.