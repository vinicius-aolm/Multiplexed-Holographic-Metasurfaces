# CLI Tools for Meta Library

## Overview / Visão Geral

Command-line tools for metasurface library processing and phase matching optimization.

Ferramentas de linha de comando para processamento de biblioteca de metassuperfície e otimização de casamento de fase.

## Available Tools / Ferramentas Disponíveis

### 1. `run_library_build.py`

Build metasurface library from Touchstone files.

**Basic Usage / Uso Básico:**
```bash
python src/cli/run_library_build.py \
  --in-dir path/to/touchstone/files \
  --experiment my_library
```

**With Recursive Search / Com Busca Recursiva:**
```bash
python src/cli/run_library_build.py \
  --in-dir library_raw \
  --recursive \
  --pattern "*.ts" \
  --out-csv library.csv \
  --out-parquet library.parquet \
  --experiment full_library
```

**Output Structure / Estrutura de Saída:**
```
results/meta_library/library_build/
└── <experiment>/
    └── <run_id>/
        ├── library_<run_id>.csv
        ├── library_<run_id>.parquet
        ├── run_meta.json
        └── README.md
```

### 2. `run_library_clean.py`

Add derived columns (amplitude, phase) to library DataFrame.

**Basic Usage / Uso Básico:**
```bash
python src/cli/run_library_clean.py \
  --in library.csv \
  --unwrap-phase \
  --phase-unit rad
```

**With Custom Column Mapping / Com Mapeamento de Colunas Personalizado:**
```bash
python src/cli/run_library_clean.py \
  --in library.parquet \
  --out-csv library_cleaned.csv \
  --unwrap-phase \
  --phase-unit deg \
  --te-real S21_real --te-imag S21_imag \
  --tm-real S12_real --tm-imag S12_imag \
  --experiment my_clean
```

**Output Structure / Estrutura de Saída:**
```
results/meta_library/library_clean/
└── <experiment>/
    └── <run_id>/
        ├── library_cleaned_<run_id>.csv
        ├── library_cleaned_<run_id>.parquet
        ├── run_meta.json
        └── README.md
```

**Added Columns / Colunas Adicionadas:**
- `S_complex_TE`, `S_complex_TM` - Complex S-parameters
- `amp_TE`, `amp_TM` - Transmission amplitudes
- `phase_TE`, `phase_TM` - Transmission phases

### 3. `run_heatmaps.py`

Generate heatmap visualizations of parameter space.

**Basic Usage / Uso Básico:**
```bash
python src/cli/run_heatmaps.py \
  --library library_cleaned.csv \
  --fields phase_TE amp_TE phase_TM amp_TM
```

**With Custom Grid and Colormap / Com Grade e Mapa de Cores Personalizados:**
```bash
python src/cli/run_heatmaps.py \
  --library library_cleaned.parquet \
  --fields phase_TE phase_TM \
  --x-col L_x --y-col L_y \
  --bins-x 150 --bins-y 150 \
  --colormap twilight \
  --dpi 300 \
  --experiment detailed_heatmaps
```

**Output Structure / Estrutura de Saída:**
```
results/meta_library/heatmaps/
└── <experiment>/
    └── <run_id>/
        ├── heatmap_phase_TE.png
        ├── heatmap_phase_TE.npy
        ├── heatmap_amp_TE.png
        ├── heatmap_amp_TE.npy
        ├── heatmap_phase_TM.png
        ├── heatmap_phase_TM.npy
        ├── heatmap_amp_TM.png
        ├── heatmap_amp_TM.npy
        ├── run_meta.json
        └── README.md
```

### 4. `run_phase_matching.py`

Perform phase matching optimization to find optimal layout.

**Basic Usage / Uso Básico:**
```bash
python src/cli/run_phase_matching.py \
  --library library_cleaned.csv \
  --target-te target_phase_te.npy \
  --target-tm target_phase_tm.npy \
  --preview
```

**With Height Filtering / Com Filtragem por Altura:**
```bash
python src/cli/run_phase_matching.py \
  --library library_cleaned.parquet \
  --target-te hologram_te.npy \
  --target-tm hologram_tm.npy \
  --use-height \
  --height-col H \
  --target-height 600 \
  --preview \
  --experiment hologram_matching
```

**Output Structure / Estrutura de Saída:**
```
results/meta_library/phase_matching/
└── <experiment>/
    └── <run_id>/
        ├── layout_lx.csv
        ├── layout_ly.csv
        ├── layout_error_map.csv
        ├── layout_summary.png
        ├── preview.png (if --preview)
        ├── run_meta.json
        └── README.md
```

## Complete Workflow Example / Exemplo de Fluxo Completo

```bash
# Step 1: Build library from Touchstone files
python src/cli/run_library_build.py \
  --in-dir library_raw \
  --recursive \
  --experiment my_metasurface

# Step 2: Clean and add derived columns
python src/cli/run_library_clean.py \
  --in results/meta_library/library_build/my_metasurface/<run_id>/library_*.csv \
  --unwrap-phase \
  --phase-unit rad \
  --experiment my_metasurface

# Step 3: Generate heatmaps for visualization
python src/cli/run_heatmaps.py \
  --library results/meta_library/library_clean/my_metasurface/<run_id>/library_cleaned_*.csv \
  --fields phase_TE amp_TE phase_TM amp_TM \
  --experiment my_metasurface

# Step 4: Perform phase matching
python src/cli/run_phase_matching.py \
  --library results/meta_library/library_clean/my_metasurface/<run_id>/library_cleaned_*.csv \
  --target-te target_phase_te.npy \
  --target-tm target_phase_tm.npy \
  --preview \
  --experiment my_metasurface
```

## Common Options / Opções Comuns

All tools support:
- `--experiment <name>` - Organize runs by experiment name
- `--out-root <path>` - Override default output root directory
- `--help` - Show detailed help for each tool

Todas as ferramentas suportam:
- `--experiment <nome>` - Organizar execuções por nome de experimento
- `--out-root <caminho>` - Substituir diretório raiz de saída padrão
- `--help` - Mostrar ajuda detalhada para cada ferramenta

## Output Conventions / Convenções de Saída

1. **Run ID / ID da Execução**: Timestamp format `YYYY-MM-DD_HH-MM-SS`
2. **Metadata / Metadados**: JSON file with run parameters and results
3. **README**: Markdown file with summary and reproducibility commands
4. **Outputs**: Data files (CSV, Parquet, NPY) and visualizations (PNG)

## Tips / Dicas

1. Use Parquet format for large libraries (faster I/O)
   Use formato Parquet para bibliotecas grandes (I/O mais rápido)

2. Enable phase unwrapping when working with continuous phase profiles
   Habilite desembrulhamento de fase ao trabalhar com perfis de fase contínuos

3. Use `--preview` flag in phase matching to verify results visually
   Use flag `--preview` no casamento de fase para verificar resultados visualmente

4. Save intermediate results to enable pipeline restart from any step
   Salve resultados intermediários para permitir reinício do pipeline de qualquer etapa

## Error Handling / Tratamento de Erros

All tools:
- Validate input files and parameters before processing
- Log detailed error messages with context
- Exit with non-zero status on failure
- Save partial results when possible

Todas as ferramentas:
- Validam arquivos de entrada e parâmetros antes do processamento
- Registram mensagens de erro detalhadas com contexto
- Saem com status diferente de zero em caso de falha
- Salvam resultados parciais quando possível

---

For module documentation, see `src/meta_library/README.md`
Para documentação dos módulos, veja `src/meta_library/README.md`
