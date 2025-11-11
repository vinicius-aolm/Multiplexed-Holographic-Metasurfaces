# Meta Library - Processamento de Bibliotecas de Metassuperfície

Conjunto de ferramentas para processar arquivos Touchstone contendo dados de parâmetros S de metassuperfícies e realizar casamento de fase para design de layouts otimizados.

## 🎯 O Que Faz

Este módulo processa dados experimentais de metassuperfícies (arquivos Touchstone) e realiza o casamento de fase necessário para conectar hologramas calculados com geometrias fabricáveis. É a ponte entre o design óptico e a fabricação.

**Funcionalidades principais:**

- Análise de arquivos Touchstone (1/2/4 portas) com extração de parâmetros S
- Cálculo de amplitude e fase de transmissão para polarizações TE/TM
- Geração de mapas de calor do espaço de parâmetros (L_x × L_y)
- Casamento de fase via KDTree para encontrar layouts ótimos

## 🚀 Uso Rápido

### Linha de Comando (CLI)

```bash
# Pipeline completo em 4 etapas

# 1. Construir biblioteca a partir de arquivos .ts
python src/cli/run_library_build.py \
  --in-dir library_raw \
  --recursive \
  --experiment minha_biblioteca

# 2. Limpar e adicionar colunas derivadas
python src/cli/run_library_clean.py \
  --in results/meta_library/library_build/minha_biblioteca/<run_id>/library_*.csv \
  --unwrap-phase \
  --experiment minha_biblioteca

# 3. Gerar mapas de calor para visualização
python src/cli/run_heatmaps.py \
  --library results/meta_library/library_clean/minha_biblioteca/<run_id>/library_cleaned_*.csv \
  --experiment minha_biblioteca

# 4. Realizar casamento de fase
python src/cli/run_phase_matching.py \
  --library results/meta_library/library_clean/minha_biblioteca/<run_id>/library_cleaned_*.csv \
  --target-te hologram_te.npy \
  --target-tm hologram_tm.npy \
  --preview \
  --experiment minha_biblioteca
```

### Como Módulo Python

```python
from meta_library import generate_df, clean_library, phase_matching
import numpy as np

# 1. Analisar arquivos Touchstone
df_raw = generate_df.touchstone_to_dataframe(
    folder="library_raw",
    recursive=True
)

# 2. Adicionar colunas derivadas
df_clean = clean_library.append_derived_columns(
    df_raw,
    unwrap_phase=True,
    phase_unit="rad"
)

# 3. Gerar mapas de calor
heatmaps = phase_matching.compute_heatmaps(
    df_clean,
    fields=("phase_TE", "amp_TE", "phase_TM", "amp_TM")
)

# 4. Realizar casamento de fase
target_te = np.load("hologram_te.npy")
target_tm = np.load("hologram_tm.npy")

layout_lx, layout_ly, error = phase_matching.perform_phase_matching(
    df_clean,
    target_phase_tm=target_tm,
    target_phase_te=target_te,
    use_height=False
)

# Salvar layouts para fabricação
np.savetxt("layout_lx.csv", layout_lx, delimiter=',')
np.savetxt("layout_ly.csv", layout_ly, delimiter=',')
```

## 📁 Estrutura de Saída

Cada ferramenta CLI cria uma pasta organizada com timestamp:

```
results/meta_library/
├── library_build/
│   └── meu_experimento/
│       └── 2024-01-15_14-30-00/
│           ├── library_2024-01-15_14-30-00.csv
│           ├── library_2024-01-15_14-30-00.parquet
│           ├── run_meta.json
│           └── README.md
├── library_clean/
│   └── meu_experimento/
│       └── 2024-01-15_14-35-00/
│           ├── library_cleaned_2024-01-15_14-35-00.csv
│           ├── library_cleaned_2024-01-15_14-35-00.parquet
│           ├── run_meta.json
│           └── README.md
├── heatmaps/
│   └── meu_experimento/
│       └── 2024-01-15_14-40-00/
│           ├── heatmap_phase_TE.png
│           ├── heatmap_phase_TE.npy
│           ├── heatmap_amp_TE.png
│           ├── heatmap_amp_TE.npy
│           ├── heatmap_phase_TM.png
│           ├── heatmap_amp_TM.png
│           ├── run_meta.json
│           └── README.md
└── phase_matching/
    └── meu_experimento/
        └── 2024-01-15_14-45-00/
            ├── layout_lx.csv
            ├── layout_ly.csv
            ├── layout_error_map.csv
            ├── layout_summary.png
            ├── preview.png (se --preview)
            ├── run_meta.json
            └── README.md
```

## 🔄 Fluxo de Trabalho

```
Arquivos Touchstone (.ts)
         ↓
   generate_df  ──→  DataFrame bruto
         ↓
   clean_library  ──→  DataFrame limpo (com amp_TE, phase_TE, etc.)
         ↓
   ├─→ phase_matching  ──→  Mapas de calor
   └─→ phase_matching  ──→  Layout de casamento de fase
```

## ⚙️ Módulos

### `generate_df.py`

Análise de arquivos Touchstone e conversão para DataFrames.

**Funções principais:**
- `parse_touchstone_params()` - Extrai parâmetros do cabeçalho
- `touchstone_to_dataframe()` - Conversão completa para DataFrame

**Colunas geradas:**
- Metadados: `arquivo`, `id_nanopilar`, `frequencia_hz`, `nports`
- Parâmetros: `L_x`, `L_y`, `H`, `Lambda` (do cabeçalho)
- Parâmetros S: `S11_real`, `S11_imag`, `S21_real`, `S21_imag`, etc.

### `clean_library.py`

Limpeza de dados e geração de colunas derivadas.

**Funções principais:**
- `append_derived_columns()` - Calcula amp_TE/TM e phase_TE/TM
- `save_library()` - Salva em CSV/Parquet

**Colunas derivadas:**
- `S_complex_TE`, `S_complex_TM` - Parâmetros S complexos
- `amp_TE`, `amp_TM` - Amplitudes de transmissão
- `phase_TE`, `phase_TM` - Fases de transmissão

**Opções:**
- Desembrulhamento de fase (por grupo de nanopilar)
- Unidade de fase: radianos ou graus
- Mapeamento customizado de colunas TE/TM

### `phase_matching.py`

Geração de mapas de calor e otimização de casamento de fase.

**Funções principais:**
- `compute_heatmaps()` - Cria grades interpoladas
- `perform_phase_matching()` - Encontra layout ótimo
- `save_heatmap_figures()` - Exporta visualizações
- `save_layout_outputs()` - Exporta resultados de layout

**Algoritmo de casamento de fase:**
1. Constrói KDTree do espaço de fase da biblioteca (phase_TE, phase_TM)
2. Para cada pixel alvo, encontra vizinho mais próximo
3. Extrai valores L_x, L_y correspondentes
4. Calcula mapa de erro RMS de fase
5. Opcional: filtra por parâmetro de altura

## 📊 Formato de Dados

### Entrada: Arquivos Touchstone

Formato de cabeçalho esperado:
```
[Number of Ports] 2
Parameters = {L_x=400; L_y=500; H=600; Lambda=1064}
```

### Saída: Estrutura do DataFrame

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `arquivo` | str | Nome do arquivo |
| `id_nanopilar` | int | ID do nanopilar |
| `frequencia_hz` | float | Frequência (Hz) |
| `L_x`, `L_y` | float | Dimensões (nm) |
| `H` | float | Altura (nm) |
| `S21_real`, `S21_imag` | float | Parâmetros S |
| `amp_TE`, `phase_TE` | float | Transmissão TE |
| `amp_TM`, `phase_TM` | float | Transmissão TM |

## 🧩 No Contexto do Repositório

Este módulo complementa as ferramentas de holografia:

- **`gs_asm.py`** - Calcula hologramas → gera mapas de fase alvo
- **`meta_library`** (este) - Faz casamento de fase → gera layouts ← *Você está aqui*
- **Fabricação** - Usa layouts L_x/L_y para produzir dispositivos

Para teoria detalhada, veja os notebooks explicativos:
- `notebooks/meta_library/01_Library_Heatmaps_Explanation.ipynb`
- `notebooks/meta_library/02_Phase_Matching_Explanation.ipynb`

## 💡 Dicas Práticas

### Formato de Arquivo

Use Parquet para bibliotecas grandes (I/O mais rápido que CSV).

### Desembrulhamento de Fase

Habilite `--unwrap-phase` ao trabalhar com perfis de fase contínuos. O desembrulhamento é feito por grupo de nanopilar.

### Filtragem por Altura

Use `--use-height` no casamento de fase para priorizar nanopilares com altura próxima ao valor alvo (tolerância de ±10%).

### Reprodutibilidade

Cada execução gera `run_meta.json` e `README.md` com comandos completos para reproduzir os resultados.

## 🔍 Para Saber Mais

- Todas as fases são em radianos por padrão (conversível para graus)
- Mapas de calor usam interpolação linear sobre espaço L_x × L_y
- Casamento de fase usa busca de vizinho mais próximo no espaço de fase
- CLIs completos documentados em `src/cli/README.md`

---

*Parte do toolkit de metassuperfícies do repositório - [Voltar ao README principal](../../README.md)*
