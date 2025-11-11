# Meta Library Module

## Overview / Visão Geral

The `meta_library` module provides tools for processing Touchstone files containing metasurface S-parameter data and performing phase matching optimization.

O módulo `meta_library` fornece ferramentas para processar arquivos Touchstone contendo dados de parâmetros S de metassuperfície e realizar otimização de casamento de fase.

## Workflow / Fluxo de Trabalho

```
Touchstone Files (.ts)
         ↓
   generate_df.py  ──→  Raw DataFrame
         ↓
   clean_library.py  ──→  Cleaned DataFrame (with amp_TE, phase_TE, etc.)
         ↓
   ├─→ phase_matching.py  ──→  Heatmaps
   └─→ phase_matching.py  ──→  Phase Matching Layout
```

## Modules / Módulos

### 1. `generate_df.py`

Parse Touchstone files and convert to structured DataFrames.

**Key Functions / Funções Principais:**
- `parse_touchstone_params(path)` - Extract header parameters
- `parse_number_of_ports_from_header(path)` - Get port count
- `touchstone_to_dataframe(folder, recursive, pattern)` - Main conversion function

**Output Columns / Colunas de Saída:**
- Metadata: `arquivo`, `caminho`, `id_nanopilar`, `frequencia_hz`, `nports`
- Parameters: `L_x`, `L_y`, `H`, `Lambda`, etc. (from file headers)
- S-parameters: `S11_real`, `S11_imag`, `S21_real`, `S21_imag`, etc.

### 2. `clean_library.py`

Add derived columns for amplitude and phase analysis.

**Key Functions / Funções Principais:**
- `append_derived_columns(df, te_cols, tm_cols, unwrap_phase, phase_unit)` - Compute derived fields
- `save_library(df, out_csv, out_parquet)` - Save processed data

**Derived Columns / Colunas Derivadas:**
- `S_complex_TE`, `S_complex_TM` - Complex S-parameters
- `amp_TE`, `amp_TM` - Transmission amplitudes
- `phase_TE`, `phase_TM` - Transmission phases

**Options / Opções:**
- Phase unwrapping (per nanopilar group)
- Phase units: radians or degrees
- Custom TE/TM column mapping

### 3. `phase_matching.py`

Generate heatmaps and perform phase matching optimization.

**Key Functions / Funções Principais:**
- `compute_heatmaps(df, x, y, fields)` - Create interpolated heatmaps
- `perform_phase_matching(df, target_phase_tm, target_phase_te, use_height)` - Find optimal layout
- `save_heatmap_figures(heatmaps, out_dir)` - Export visualizations
- `save_layout_outputs(layout_lx, layout_ly, error_map, out_dir)` - Export layout results

**Phase Matching Algorithm / Algoritmo de Casamento de Fase:**
1. Build KDTree from library phase space (phase_TE, phase_TM)
2. For each target pixel, find nearest neighbor in library
3. Extract corresponding L_x, L_y values
4. Compute RMS phase error map
5. Optional: filter by height parameter

## Usage Examples / Exemplos de Uso

### Python API

```python
from meta_library import generate_df, clean_library, phase_matching

# 1. Parse Touchstone files
df_raw = generate_df.touchstone_to_dataframe("path/to/touchstone/files")

# 2. Add derived columns
df_clean = clean_library.append_derived_columns(
    df_raw, unwrap_phase=True, phase_unit="rad"
)
clean_library.save_library(df_clean, out_csv="library.csv")

# 3. Generate heatmaps
heatmaps = phase_matching.compute_heatmaps(
    df_clean, fields=("phase_TE", "amp_TE", "phase_TM", "amp_TM")
)

# 4. Perform phase matching
target_te = np.load("target_phase_te.npy")
target_tm = np.load("target_phase_tm.npy")
layout_lx, layout_ly, error = phase_matching.perform_phase_matching(
    df_clean, target_tm, target_te
)
```

### Command Line

See `src/cli/README.md` for CLI usage examples.

Veja `src/cli/README.md` para exemplos de uso CLI.

## Data Format / Formato de Dados

### Input: Touchstone Files

Expected header format:
```
[Number of Ports] 2
Parameters = {L_x=400; L_y=500; H=600; Lambda=1064}
```

### Output: DataFrame Structure

| Column | Type | Description |
|--------|------|-------------|
| `arquivo` | str | Filename |
| `id_nanopilar` | int | Nanopilar ID |
| `frequencia_hz` | float | Frequency (Hz) |
| `L_x`, `L_y` | float | Dimensions (nm) |
| `H` | float | Height (nm) |
| `S21_real`, `S21_imag` | float | S-parameters |
| `amp_TE`, `phase_TE` | float | TE transmission |
| `amp_TM`, `phase_TM` | float | TM transmission |

## Dependencies / Dependências

- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scipy` - Interpolation and spatial search
- `matplotlib` - Visualization
- `scikit-rf` - Touchstone file parsing

## Notes / Notas

- All phases are in radians by default (can convert to degrees)
- Phase unwrapping is performed per nanopilar group
- Heatmaps use linear interpolation over L_x × L_y space
- Phase matching uses nearest-neighbor search in phase space
- Height filtering uses ±10% tolerance around target value

---

For CLI tools and notebooks, see:
- `src/cli/README.md`
- `notebooks/meta_library/README.md`
