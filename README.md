# Multiplexed Holographic Metasurfaces

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

> Capstone project at Ilum - School of Science, Brazilian Center for Research in Energy and Materials (CNPEM)

A comprehensive toolkit for designing and simulating multiplexed holographic metasurfaces. This project implements computational methods for phase-only hologram generation and metasurface design, bridging optical design with nanofabrication.

**📚 [Quick Start Guide](docs/QUICKSTART.md)** | **📖 [Full Documentation](docs/)** | **🏗️ [Structure](docs/STRUCTURE.md)** | **🤝 [Contributing](CONTRIBUTING.md)**

## 🎯 Overview

This repository contains tools for:

- **Hologram Generation**: Gerchberg-Saxton algorithm with Angular Spectrum Method (GS+ASM)
- **Dammann Gratings**: Periodic metasurface design for uniform spot arrays
- **Meta-Atom Library Processing**: Analysis and optimization of metasurface building blocks
- **Phase Matching**: Connecting holographic designs to fabricable geometries

## 📁 Repository Structure

```
.
├── src/                    # Source code modules
│   ├── cli/               # Command-line interface tools
│   ├── holography/        # GS+ASM hologram generation
│   ├── dammann/           # Dammann grating generation
│   ├── meta_library/      # Metasurface library processing
│   ├── optimization/      # Optimization algorithms (GA, PSO, CPPN)
│   ├── simulation/        # Electromagnetic simulation tools
│   └── utils/             # Shared utilities
├── notebooks/             # Jupyter notebooks for exploration
│   ├── holography/        # Holography demonstrations
│   ├── meta_library/      # Library processing tutorials
│   ├── optimization/      # Optimization studies
│   └── legacy_exploration/# Historical exploration notebooks
├── data/                  # Data directory
│   ├── raw/              # Raw data files (Touchstone, MAT files)
│   ├── processed/        # Processed data
│   ├── targets/          # Target images for holography
│   ├── meta_library/     # Metasurface library data
│   └── models/           # Trained ML models
├── results/              # Organized output directory
│   ├── holography/       # Hologram generation results
│   ├── holography-dammann/ # Dammann grating results
│   ├── meta_library/     # Library processing outputs
│   ├── optimization/     # Optimization run results
│   └── simulation/       # Simulation outputs
├── tests/                # Test suite
├── docs/                 # Documentation
│   ├── figures/          # Figures and diagrams
│   └── references/       # Reference papers and reports
├── animations/           # GS algorithm animations
├── macroCST/            # CST Studio macros
├── scripts/             # Standalone scripts
│   └── legacy/          # Legacy scripts (preserved for reference)
├── hooks/               # Git hooks
└── legacy/              # Legacy code (frozen for reference)
    └── phase_matching/  # Original phase matching code

```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vinicius-aolm/Multiplexed-Holographic-Metasurfaces.git
cd Multiplexed-Holographic-Metasurfaces

# Install dependencies (create requirements.txt if needed)
pip install numpy scipy matplotlib pillow pandas scikit-learn jupyter
```

### Basic Usage

#### Generate a Hologram

```bash
python src/holography/gs_asm.py \
  --targets data/targets/common/ilum.png data/targets/common/ufabc.png \
  --experiment my_hologram \
  --iters 200
```

#### Process Metasurface Library

```bash
# Build library from Touchstone files
python src/cli/run_library_build.py \
  --in-dir data/raw/touchstone \
  --recursive

# Clean and add derived columns
python src/cli/run_library_clean.py \
  --in results/meta_library/library_build/<run_id>/library_*.csv \
  --unwrap-phase

# Generate heatmaps
python src/cli/run_heatmaps.py \
  --library results/meta_library/library_clean/<run_id>/library_cleaned_*.csv

# Perform phase matching
python src/cli/run_phase_matching.py \
  --library results/meta_library/library_clean/<run_id>/library_cleaned_*.csv \
  --target-te hologram_te.npy \
  --target-tm hologram_tm.npy
```

### Python API Usage

```python
from src.holography.gs_asm import run_batch
from src.meta_library import generate_df, clean_library, phase_matching
from pathlib import Path

# Generate hologram
targets = [("ilum", Path("data/targets/common/ilum.png"))]
result = run_batch(targets=targets, experiment="test", pol_label="X")

# Process metasurface library
df_raw = generate_df.touchstone_to_dataframe("data/raw/touchstone", recursive=True)
df_clean = clean_library.append_derived_columns(df_raw, unwrap_phase=True)
```

## 📚 Documentation

Detailed documentation is available in each module:

- **CLI Tools**: [`src/cli/README.md`](src/cli/README.md)
- **Holography**: [`src/holography/README.md`](src/holography/README.md)
- **Dammann Gratings**: [`src/dammann/README.md`](src/dammann/README.md)
- **Meta Library**: [`src/meta_library/README.md`](src/meta_library/README.md)

Interactive tutorials are available in Jupyter notebooks:

- **Library Processing**: [`notebooks/meta_library/`](notebooks/meta_library/)
- **Holography**: [`notebooks/holography/`](notebooks/holography/)
- **Optimization**: [`notebooks/optimization/`](notebooks/optimization/)

## 🔬 Features

### Hologram Generation

- **GS+ASM**: Gerchberg-Saxton algorithm with Angular Spectrum propagation
- **Dammann Gratings**: Uniform spot array generation
- **Multi-target support**: Process multiple images in batch
- **Automatic visualization**: Convergence plots, phase maps, reconstructions

### Metasurface Library Processing

- **Touchstone parsing**: Extract S-parameters from measurement data
- **Phase/amplitude derivation**: Calculate TE/TM transmission properties
- **Heatmap generation**: Visualize parameter space coverage
- **Phase matching**: Find optimal nanopillar geometries for target phases

### Optimization

- **Genetic Algorithms (GA)**: Evolutionary optimization
- **Particle Swarm Optimization (PSO)**: Swarm-based optimization
- **CPPN (Compositional Pattern Producing Networks)**: Neural network-based design
- **Surrogate Models**: Machine learning for fast evaluation

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_meta_library.py
```

## 📊 Output Organization

All tools generate organized outputs with:

- Timestamped run directories
- Comprehensive metadata (`run_meta.json`)
- Auto-generated READMEs for reproducibility
- Both CSV and Parquet formats (where applicable)

Example output structure:
```
results/holography/gs_x/my_experiment/2024-01-15_14-30-00/
├── ilum/
│   ├── phase_map__*.txt
│   ├── mapa_de_fase__*.png
│   ├── reconstruida__*.png
│   └── convergencia__*.png
├── ufabc/
│   └── ...
├── summary__*.png
└── run_meta.json
```

## 🤝 Contributing

This project follows the structure described in the associated monograph. When contributing:

1. Maintain the established directory structure
2. Follow existing code style and documentation patterns
3. Add tests for new functionality
4. Update relevant README files
5. Use the CLI patterns for consistency

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{multiplexed-holographic-metasurfaces,
  author = {Vinícius, João, Humberto},
  title = {Multiplexed Holographic Metasurfaces},
  year = {2024},
  institution = {Ilum - School of Science, CNPEM},
  note = {Capstone Project}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 References

Key references are available in [`docs/references/`](docs/references/):

- Project reports and monographs
- Related papers on metasurfaces and holography
- Technical documentation

## 👥 Authors

- **Vinícius**
- **João** 
- **Humberto**
- **Gabriel**

*Capstone Project - Ilum - School of Science, CNPEM*

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: Legacy code is preserved in the `legacy/` directory for reference but is not actively maintained. All active development occurs in `src/`.
