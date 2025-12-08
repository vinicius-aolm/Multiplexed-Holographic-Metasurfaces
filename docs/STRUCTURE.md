# Directory Structure Documentation

This document provides detailed information about the repository organization.

## 📁 Top-Level Structure

```
Multiplexed-Holographic-Metasurfaces/
├── .git/                    # Git version control
├── .gitattributes          # Git attributes configuration
├── .gitignore              # Git ignore patterns
├── LICENSE                 # Apache 2.0 License
├── README.md              # Main project documentation
├── CONTRIBUTING.md        # Contribution guidelines
├── CHANGELOG.md           # Version history and changes
├── requirements.txt       # Python dependencies
├── src/                   # Source code (production)
├── notebooks/             # Jupyter notebooks (tutorials/exploration)
├── data/                  # Data files and datasets
├── results/               # Generated outputs
├── tests/                 # Test suite
├── docs/                  # Documentation and references
├── scripts/               # Standalone scripts
├── animations/            # GS algorithm animation files
├── macroCST/             # CST Studio macro files
├── hooks/                # Git hooks
└── legacy/               # Legacy code (frozen)
```

## 🔧 Source Code (`src/`)

Production-quality code organized by functionality.

```
src/
├── cli/                   # Command-line interface tools
│   ├── run_library_build.py      # Build library from Touchstone
│   ├── run_library_clean.py      # Clean and derive columns
│   ├── run_heatmaps.py           # Generate parameter space heatmaps
│   ├── run_phase_matching.py     # Perform phase matching
│   └── run_surrogate_mlp.py      # Train ML surrogate models
├── holography/            # Hologram generation
│   └── gs_asm.py                 # GS+ASM algorithm implementation
├── dammann/              # Dammann grating generation
│   └── dammann_fft.py            # FFT-based Dammann grating
├── meta_library/         # Metasurface library processing
│   ├── generate_df.py            # Touchstone parsing
│   ├── clean_library.py          # Data cleaning utilities
│   ├── phase_matching.py         # Phase matching algorithms
│   └── ml/                       # Machine learning models
│       └── surrogate_mlp.py      # MLP surrogate model
├── optimization/         # Optimization algorithms
│   └── (GA, PSO, CPPN implementations)
├── simulation/          # Electromagnetic simulation tools
│   └── (Simulation-related code)
└── utils/               # Shared utilities
    └── (Common helper functions)
```

**Key Principles:**
- Each module has a README.md explaining its purpose
- Code is well-documented with docstrings
- CLI tools follow consistent patterns
- All imports use absolute paths from `src/`

## 📓 Notebooks (`notebooks/`)

Interactive Jupyter notebooks for exploration and tutorials.

```
notebooks/
├── holography/           # Holography demonstrations
│   ├── explanations/           # Educational notebooks
│   └── (working notebooks)
├── meta_library/        # Library processing tutorials
│   ├── 01_Library_Heatmaps_Explanation.ipynb
│   ├── 02_Phase_Matching_Explanation.ipynb
│   └── README.md              # Notebook documentation
├── optimization/        # Optimization studies
│   ├── optimization_ga.ipynb
│   ├── optimization_pso.ipynb
│   └── (CPPN and other studies)
└── legacy_exploration/  # Historical exploration notebooks
    └── (archived notebooks)
```

**Key Principles:**
- Notebooks are bilingual (English/Portuguese)
- Each notebook includes reproducibility section
- Notebooks demonstrate both interactive use and equivalent CLI commands
- Clear markdown explanations with code examples

## 💾 Data (`data/`)

Data files, organized by type and processing stage.

```
data/
├── raw/                  # Raw, unprocessed data
│   ├── (Touchstone files - not committed)
│   ├── chosen_indices.mat
│   └── tx_ty.mat
├── processed/           # Processed, cleaned data
│   └── (CSV/Parquet files - not committed)
├── targets/            # Target images for holography
│   └── common/
│       ├── ilum.png
│       ├── ufabc.png
│       └── espaco.jpeg
├── meta_library/       # Metasurface library data
│   └── (Library files - not committed)
└── models/             # Trained ML models
    └── (Model checkpoints - not committed)
```

**Key Principles:**
- Raw data preserved as-is (when size permits)
- Processed data regenerated from raw
- Target images version controlled (small files)
- Large datasets (.ts, .csv, .parquet) excluded via .gitignore

## 📊 Results (`results/`)

Generated outputs from all tools, organized by tool and experiment.

```
results/
├── holography/          # Hologram generation outputs
│   ├── gs_x/                    # X polarization
│   └── gs_y/                    # Y polarization
├── holography-dammann/ # Dammann grating outputs
│   ├── dammann/
│   ├── gs_x/
│   └── gs_y/
├── meta_library/       # Library processing outputs
│   ├── library_build/
│   ├── library_clean/
│   ├── heatmaps/
│   └── phase_matching/
├── optimization/       # Optimization run results
│   └── (GA, PSO, CPPN results)
└── simulation/         # Simulation outputs
    └── (Simulation results)
```

**Structure Pattern:**
```
results/<tool>/<experiment>/<timestamp>/
├── <output_files>
├── run_meta.json       # Metadata for reproducibility
└── README.md          # Auto-generated documentation
```

**Key Principles:**
- All outputs timestamped with ISO format
- Each run self-documented
- Results directory typically not committed (regenerated)
- Structure enables easy comparison between runs

## 🧪 Tests (`tests/`)

Test suite for validating functionality.

```
tests/
├── test_meta_library.py
└── (additional test files)
```

**Key Principles:**
- Tests organized by module
- Use pytest framework
- Test both success and error cases
- Keep tests independent

## 📚 Documentation (`docs/`)

Additional documentation and reference materials.

```
docs/
├── figures/             # Figures and diagrams
│   └── (visualization assets)
└── references/          # Reference papers and reports
    ├── Projeto_Final_Vinicius_Joao_Humberto (7).pdf
    ├── TCC_Relatório_acompanhamento__Version_11_ (4) (2).pdf
    └── (other papers and reports)
```

**Key Principles:**
- Figures used in documentation
- References to academic work
- Project reports and monographs

## 🎬 Animations (`animations/`)

Educational animations demonstrating the GS algorithm.

```
animations/
├── GS_Animation_X_1.py
├── GS_Animation_X_2.py
├── GS_Animation_X_3.py
├── GS_Animation_Y_1.py
├── GS_Animation_Y_2.py
├── GS_Step1_Final_Layout.gif
├── GS_Step2_Forward_Final_Spaced.gif
└── (other animation assets)
```

**Purpose:** Visual demonstrations of algorithm convergence and physics.

## ⚙️ Scripts (`scripts/`)

Standalone scripts for various tasks.

```
scripts/
├── README.md
└── legacy/              # Legacy scripts (preserved)
    ├── analisa_s4p_folgas.py
    ├── malha_local.py
    └── pipeline_metaholo_auto.m
```

**Key Principles:**
- Legacy scripts preserved but not actively maintained
- Standalone utilities that don't fit in main modules
- Documented in scripts/README.md

## 🏛️ Legacy (`legacy/`)

Historical code frozen for reference.

```
legacy/
└── phase_matching/      # Original phase matching implementation
    └── README.md
```

**Purpose:** Preserve original implementations as reference without active maintenance.

## 🎨 Other Directories

### `macroCST/`
CST Studio macro files for electromagnetic simulation.

### `hooks/`
Git hooks for automation (currently contains .gitkeep).

## 🔄 Workflow

Typical data flow through the repository:

```
1. Raw Data → src/meta_library → Processed Data
2. Target Images → src/holography → Phase Maps
3. Phase Maps + Library → src/meta_library/phase_matching → Layouts
4. All outputs → results/<organized_structure>
```

## 📝 Best Practices

1. **Keep root clean**: Only configuration and documentation at top level
2. **Organize by function**: Related code stays together
3. **Document everything**: Each directory has README or documentation
4. **Timestamp outputs**: All results include timestamps and metadata
5. **Preserve history**: Legacy code in dedicated directories

## 🔗 Navigation

- **For code**: Start in `src/`
- **For learning**: Start in `notebooks/`
- **For data**: Check `data/` subdirectories
- **For results**: Browse `results/` by tool/experiment
- **For references**: See `docs/references/`

---

*This structure follows the organization described in the project monograph and supports reproducible computational research.*
