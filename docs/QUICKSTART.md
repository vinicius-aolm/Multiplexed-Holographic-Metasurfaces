# Quick Start Guide

Get up and running with Multiplexed Holographic Metasurfaces in minutes!

## ⚡ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vinicius-aolm/Multiplexed-Holographic-Metasurfaces.git
cd Multiplexed-Holographic-Metasurfaces
```

### 2. Install Dependencies

#### Using pip

```bash
pip install -r requirements.txt
```

#### Using conda (recommended)

```bash
conda create -n metasurfaces python=3.9
conda activate metasurfaces
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import numpy, scipy, matplotlib, pandas; print('Dependencies OK!')"
```

## 🚀 Your First Hologram

### Command Line

```bash
# Navigate to repository
cd Multiplexed-Holographic-Metasurfaces

# Generate a simple hologram
python src/holography/gs_asm.py \
  --targets data/targets/common/ilum.png \
  --experiment my_first_hologram \
  --iters 50
```

**Output:** Check `results/holography/gs_x/my_first_hologram/<timestamp>/`

### Python Script

Create a file `test_hologram.py`:

```python
from pathlib import Path
from src.holography.gs_asm import run_batch

# Configure
targets = [("test", Path("data/targets/common/ilum.png"))]

# Run
result = run_batch(
    targets=targets,
    out_root=Path("results/holography/gs_x"),
    experiment="my_first_hologram",
    pol_label="X",
    num_iter=50
)

print(f"✅ Done! Results saved to: {result}")
```

Run it:
```bash
python test_hologram.py
```

## 📚 Next Steps

### Explore Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/
```

**Recommended order:**
1. `notebooks/meta_library/01_Library_Heatmaps_Explanation.ipynb`
2. `notebooks/meta_library/02_Phase_Matching_Explanation.ipynb`
3. `notebooks/holography/` (various examples)

### Try Different Algorithms

#### Dammann Gratings

```bash
python src/dammann/dammann_fft.py \
  --experiment dammann_test \
  --iters 200
```

#### Metasurface Library Processing

```bash
# Note: Requires Touchstone data files in data/raw/
python src/cli/run_library_build.py \
  --in-dir data/raw/touchstone \
  --recursive

# Clean the library
python src/cli/run_library_clean.py \
  --in results/meta_library/library_build/<experiment>/<timestamp>/library_*.csv \
  --unwrap-phase
```

## 🛠️ Common Tasks

### View Available CLI Options

```bash
python src/holography/gs_asm.py --help
python src/dammann/dammann_fft.py --help
python src/cli/run_library_build.py --help
```

### Check Results

All results are organized under `results/` with structure:
```
results/<tool>/<experiment>/<timestamp>/
├── output_files
├── run_meta.json    # Complete metadata
└── README.md       # Auto-generated documentation
```

### Use Your Own Target Images

```bash
# Place your image in data/targets/
cp my_image.png data/targets/common/

# Generate hologram
python src/holography/gs_asm.py \
  --targets data/targets/common/my_image.png \
  --experiment custom_hologram
```

**Image Requirements:**
- Format: PNG, JPEG, or common image formats
- Will be converted to grayscale
- Will be resized to 450×450 pixels by default

## 📖 Learn More

### Documentation

- **Main README**: [`README.md`](../README.md) - Project overview
- **Module Docs**: Each module in `src/` has detailed README
- **Structure**: [`docs/STRUCTURE.md`](STRUCTURE.md) - Repository organization
- **Contributing**: [`CONTRIBUTING.md`](../CONTRIBUTING.md) - Development guide

### Tutorials

- **Meta Library Processing**: `notebooks/meta_library/`
- **Holography**: `notebooks/holography/`
- **Optimization**: `notebooks/optimization/`

## 🐛 Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
```

Or run from repository root:
```bash
cd /path/to/Multiplexed-Holographic-Metasurfaces
python src/holography/gs_asm.py ...
```

### Missing Dependencies

```bash
# Install specific package
pip install <package-name>

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

### Out of Memory

Reduce problem size:
```bash
# Reduce iterations
python src/holography/gs_asm.py --iters 50 ...

# Or use smaller target images
```

### File Not Found

Check paths:
```bash
# Verify file exists
ls -la data/targets/common/ilum.png

# Use absolute path if needed
python src/holography/gs_asm.py \
  --targets /absolute/path/to/image.png ...
```

## 💡 Tips

1. **Start small**: Use fewer iterations (50-100) for quick tests
2. **Check metadata**: Every run creates `run_meta.json` with complete info
3. **Use timestamps**: Outputs are timestamped - no overwrites!
4. **Read READMEs**: Each tool auto-generates README with reproduction command
5. **Explore notebooks**: Interactive learning is easier than CLI

## 🎯 Quick Reference

### Common Commands

```bash
# Hologram (X polarization)
python src/holography/gs_asm.py --targets <image> --experiment <name>

# Dammann grating
python src/dammann/dammann_fft.py --experiment <name>

# Build library
python src/cli/run_library_build.py --in-dir <dir> --recursive

# Clean library
python src/cli/run_library_clean.py --in <file> --unwrap-phase

# Generate heatmaps
python src/cli/run_heatmaps.py --library <file>

# Phase matching
python src/cli/run_phase_matching.py \
  --library <file> \
  --target-te <te_phase.npy> \
  --target-tm <tm_phase.npy>
```

### Python API

```python
# Holography
from src.holography.gs_asm import run_batch
result = run_batch(targets=[("name", Path("image.png"))], ...)

# Library processing
from src.meta_library import generate_df, clean_library, phase_matching
df = generate_df.touchstone_to_dataframe("folder/")
df_clean = clean_library.append_derived_columns(df)
heatmaps = phase_matching.compute_heatmaps(df_clean)
```

## ❓ Need Help?

- **Read the docs**: Start with module READMEs in `src/`
- **Check notebooks**: Interactive examples in `notebooks/`
- **Open an issue**: [GitHub Issues](https://github.com/vinicius-aolm/Multiplexed-Holographic-Metasurfaces/issues)
- **See examples**: Check `run_meta.json` in `results/` for working examples

---

**Ready to go deeper?** Check out [`CONTRIBUTING.md`](../CONTRIBUTING.md) for development setup!
