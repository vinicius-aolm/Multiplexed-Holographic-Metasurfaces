# CLIs Meta Library - Ferramentas de Linha de Comando

Ferramentas de linha de comando para processar bibliotecas de metassuperfície: construção, limpeza, visualização e casamento de fase.

## 🎯 Visão Geral

Quatro CLIs que implementam o pipeline completo de processamento:

1. **`run_library_build.py`** - Analisar arquivos Touchstone → gerar DataFrame
2. **`run_library_clean.py`** - Adicionar colunas derivadas (amp/fase TE/TM)
3. **`run_heatmaps.py`** - Visualizar espaço de parâmetros
4. **`run_phase_matching.py`** - Encontrar layouts ótimos

## 🚀 Início Rápido

```bash
# Pipeline completo
python src/cli/run_library_build.py --in-dir library_raw --recursive
python src/cli/run_library_clean.py --in results/meta_library/library_build/library_build/<run_id>/library_*.csv --unwrap-phase
python src/cli/run_heatmaps.py --library results/meta_library/library_clean/library_clean/<run_id>/library_cleaned_*.csv
python src/cli/run_phase_matching.py --library results/meta_library/library_clean/library_clean/<run_id>/library_cleaned_*.csv --target-te fase_te.npy --target-tm fase_tm.npy
```

---

## 1. run_library_build.py

### 📝 Descrição

Constrói biblioteca estruturada a partir de arquivos Touchstone (.ts). Analisa parâmetros S de múltiplas frequências e extrai metadados do cabeçalho.

### ⚙️ Argumentos

| Argumento | Obrigatório | Padrão | Descrição |
|-----------|-------------|--------|-----------|
| `--in-dir` | ✅ | - | Diretório com arquivos Touchstone |
| `--recursive` | - | False | Buscar subdiretórios |
| `--pattern` | - | `*.ts` | Padrão de arquivo |
| `--out-csv` | - | Auto | Caminho CSV de saída |
| `--out-parquet` | - | Auto | Caminho Parquet de saída |
| `--experiment` | - | `library_build` | Nome do experimento |
| `--out-root` | - | Auto | Diretório raiz de saída |

### 📤 Saídas

```
results/meta_library/library_build/<experiment>/<run_id>/
├── library_<run_id>.csv
├── library_<run_id>.parquet
├── run_meta.json
└── README.md
```

### 💡 Exemplos

```bash
# Básico
python src/cli/run_library_build.py --in-dir data/touchstone

# Recursivo com experimento nomeado
python src/cli/run_library_build.py \
  --in-dir library_raw \
  --recursive \
  --experiment biblioteca_v2

# Saída customizada
python src/cli/run_library_build.py \
  --in-dir library_raw \
  --out-csv minha_biblioteca.csv \
  --out-parquet minha_biblioteca.parquet
```

---

## 2. run_library_clean.py

### 📝 Descrição

Limpa biblioteca e adiciona colunas derivadas de amplitude e fase para polarizações TE e TM. Suporta desembrulhamento de fase e conversão de unidades.

### ⚙️ Argumentos

| Argumento | Obrigatório | Padrão | Descrição |
|-----------|-------------|--------|-----------|
| `--in` | ✅ | - | Arquivo de entrada (CSV/Parquet) |
| `--out-csv` | - | Auto | Caminho CSV de saída |
| `--out-parquet` | - | Auto | Caminho Parquet de saída |
| `--unwrap-phase` | - | False | Desembrulhar fase |
| `--phase-unit` | - | `rad` | Unidade: `rad` ou `deg` |
| `--te-real` | - | `S21_real` | Coluna real TE |
| `--te-imag` | - | `S21_imag` | Coluna imaginária TE |
| `--tm-real` | - | `S12_real` | Coluna real TM |
| `--tm-imag` | - | `S12_imag` | Coluna imaginária TM |
| `--experiment` | - | `library_clean` | Nome do experimento |
| `--out-root` | - | Auto | Diretório raiz de saída |

### 📤 Saídas

```
results/meta_library/library_clean/<experiment>/<run_id>/
├── library_cleaned_<run_id>.csv
├── library_cleaned_<run_id>.parquet
├── run_meta.json
└── README.md
```

**Colunas adicionadas:**
- `S_complex_TE`, `S_complex_TM` - Parâmetros S complexos
- `amp_TE`, `amp_TM` - Amplitudes
- `phase_TE`, `phase_TM` - Fases

### 💡 Exemplos

```bash
# Básico
python src/cli/run_library_clean.py --in library.csv

# Com desembrulhamento e graus
python src/cli/run_library_clean.py \
  --in library.csv \
  --unwrap-phase \
  --phase-unit deg

# Mapeamento customizado de colunas
python src/cli/run_library_clean.py \
  --in library.csv \
  --te-real S31_real --te-imag S31_imag \
  --tm-real S41_real --tm-imag S41_imag
```

---

## 3. run_heatmaps.py

### 📝 Descrição

Gera mapas de calor 2D do espaço de parâmetros (L_x × L_y) para amplitude e fase TE/TM. Usa interpolação linear para criar grades regulares.

### ⚙️ Argumentos

| Argumento | Obrigatório | Padrão | Descrição |
|-----------|-------------|--------|-----------|
| `--library` | ✅ | - | Arquivo de biblioteca limpa |
| `--out-dir` | - | Auto | Diretório de saída |
| `--fields` | - | Todos | Campos para mapas (sep. espaço) |
| `--bins-x` | - | 100 | Bins na direção x |
| `--bins-y` | - | 100 | Bins na direção y |
| `--colormap` | - | `viridis` | Mapa de cores Matplotlib |
| `--dpi` | - | 300 | Resolução (DPI) |
| `--experiment` | - | `heatmaps` | Nome do experimento |
| `--out-root` | - | Auto | Diretório raiz de saída |

### 📤 Saídas

```
results/meta_library/heatmaps/<experiment>/<run_id>/
├── heatmap_phase_TE.png
├── heatmap_phase_TE.npy
├── heatmap_amp_TE.png
├── heatmap_amp_TE.npy
├── heatmap_phase_TM.png
├── heatmap_amp_TM.png
├── run_meta.json
└── README.md
```

### 💡 Exemplos

```bash
# Básico (todos os campos)
python src/cli/run_heatmaps.py --library library_cleaned.csv

# Apenas fase
python src/cli/run_heatmaps.py \
  --library library_cleaned.csv \
  --fields phase_TE phase_TM

# Alta resolução customizada
python src/cli/run_heatmaps.py \
  --library library_cleaned.csv \
  --bins-x 200 --bins-y 200 \
  --dpi 600 \
  --colormap plasma
```

---

## 4. run_phase_matching.py

### 📝 Descrição

Realiza casamento de fase entre biblioteca e mapas de fase alvo (de hologramas). Para cada pixel, encontra o nanopilar com fase TE/TM mais próxima usando KDTree.

### ⚙️ Argumentos

| Argumento | Obrigatório | Padrão | Descrição |
|-----------|-------------|--------|-----------|
| `--library` | ✅ | - | Arquivo de biblioteca limpa |
| `--target-te` | ✅ | - | Arquivo com fase TE alvo (.npy/.npz) |
| `--target-tm` | ✅ | - | Arquivo com fase TM alvo (.npy/.npz) |
| `--use-height` | - | False | Filtrar por altura |
| `--height-col` | - | `H` | Coluna de altura |
| `--target-height` | - | Auto | Valor alvo de altura |
| `--out-dir` | - | Auto | Diretório de saída |
| `--preview` | - | False | Gerar figura de preview |
| `--experiment` | - | `phase_matching` | Nome do experimento |
| `--out-root` | - | Auto | Diretório raiz de saída |

### 📤 Saídas

```
results/meta_library/phase_matching/<experiment>/<run_id>/
├── layout_lx.csv          # Valores L_x por pixel
├── layout_ly.csv          # Valores L_y por pixel
├── layout_error_map.csv   # Erro RMS por pixel
├── layout_summary.png     # Visualização dos 3 mapas
├── preview.png            # (se --preview) Comparação fases
├── run_meta.json
└── README.md
```

### 💡 Exemplos

```bash
# Básico
python src/cli/run_phase_matching.py \
  --library library_cleaned.csv \
  --target-te holograma_te.npy \
  --target-tm holograma_tm.npy

# Com filtragem por altura e preview
python src/cli/run_phase_matching.py \
  --library library_cleaned.csv \
  --target-te holograma_te.npy \
  --target-tm holograma_tm.npy \
  --use-height \
  --target-height 600 \
  --preview

# Saída customizada
python src/cli/run_phase_matching.py \
  --library library_cleaned.csv \
  --target-te holograma_te.npy \
  --target-tm holograma_tm.npy \
  --out-dir meus_layouts \
  --experiment design_final
```

---

## 📁 Estrutura de Saída Geral

Todas as ferramentas seguem o padrão:

```
results/meta_library/<ferramenta>/<experiment>/<run_id>/
├── <arquivos_especificos>
├── run_meta.json    # Metadados completos (reprodutibilidade)
└── README.md        # Documentação da execução
```

**run_id** = timestamp no formato `YYYY-MM-DD_HH-MM-SS`

## 🔄 Pipeline Típico

```bash
#!/bin/bash
# Script de exemplo do pipeline completo

EXP="meu_experimento"

# 1. Construir biblioteca
python src/cli/run_library_build.py \
  --in-dir library_raw \
  --recursive \
  --experiment $EXP

# 2. Obter run_id da etapa anterior
BUILD_RUN=$(ls -t results/meta_library/library_build/$EXP/ | head -1)
LIBRARY="results/meta_library/library_build/$EXP/$BUILD_RUN/library_*.csv"

# 3. Limpar
python src/cli/run_library_clean.py \
  --in $LIBRARY \
  --unwrap-phase \
  --experiment $EXP

# 4. Obter biblioteca limpa
CLEAN_RUN=$(ls -t results/meta_library/library_clean/$EXP/ | head -1)
CLEAN_LIB="results/meta_library/library_clean/$EXP/$CLEAN_RUN/library_cleaned_*.csv"

# 5. Heatmaps
python src/cli/run_heatmaps.py \
  --library $CLEAN_LIB \
  --experiment $EXP

# 6. Casamento de fase
python src/cli/run_phase_matching.py \
  --library $CLEAN_LIB \
  --target-te hologram_te.npy \
  --target-tm hologram_tm.npy \
  --preview \
  --experiment $EXP

echo "Pipeline completo!"
```

## 💡 Dicas

### Reprodutibilidade

Cada execução gera `run_meta.json` com comando completo para reproduzir:

```bash
cat results/meta_library/<ferramenta>/<exp>/<run_id>/README.md
# Seção "Reprodutibilidade" contém o comando exato usado
```

### Encadeamento com `find`

```bash
# Processar automaticamente o output mais recente
python src/cli/run_library_clean.py \
  --in $(find results/meta_library/library_build -name "*.csv" | head -1)
```

### Formato de Arquivo

- Use **CSV** para inspeção manual e compatibilidade
- Use **Parquet** para bibliotecas grandes (I/O 10x mais rápido)

### Integração com Holografia

Mapas de fase alvo geralmente vêm de:
```bash
python src/holography/gs_asm.py --targets imagem.png --experiment holo
# Saída: phase_map__*.txt (converter para .npy)
```

## 🔍 Troubleshooting

### "Colunas necessárias faltando"

Verifique mapeamento de colunas com `--te-real`, `--te-imag`, etc.

### "Nenhum arquivo correspondendo ao padrão"

Use `--pattern "*.s2p"` se arquivos têm extensão diferente.

### "Formas das fases alvo devem corresponder"

Arrays TE e TM devem ter mesma dimensão (rows × cols).

### Performance lenta no casamento de fase

Use `--use-height` para reduzir espaço de busca.

---

*Parte do toolkit de metassuperfícies - [Voltar ao README do módulo](../meta_library/README.md)*
