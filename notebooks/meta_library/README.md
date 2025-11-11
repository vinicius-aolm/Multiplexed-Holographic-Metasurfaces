# Meta Library Notebooks

## Overview / Visão Geral

Educational Jupyter notebooks demonstrating metasurface library processing and phase matching workflows.

Notebooks Jupyter educacionais demonstrando processamento de biblioteca de metassuperfície e fluxos de trabalho de casamento de fase.

## Available Notebooks / Notebooks Disponíveis

### 01_Library_Heatmaps_Explanation.ipynb

**Purpose / Propósito:**
- Demonstrates heatmap generation and visualization
- Demonstra geração e visualização de mapas de calor
- Explores parameter space coverage
- Explora cobertura do espaço de parâmetros

**Topics Covered / Tópicos Cobertos:**
- Loading cleaned library data / Carregamento de dados de biblioteca limpa
- Computing interpolated heatmaps / Computação de mapas de calor interpolados
- Visualizing amplitude and phase distributions / Visualização de distribuições de amplitude e fase
- Interpreting parameter space / Interpretação do espaço de parâmetros

**Prerequisites / Pré-requisitos:**
- Cleaned library file with derived columns (from `run_library_clean.py`)
- Arquivo de biblioteca limpa com colunas derivadas (de `run_library_clean.py`)

**Estimated Runtime / Tempo Estimado:**
- 2-5 minutes depending on library size
- 2-5 minutos dependendo do tamanho da biblioteca

### 02_Phase_Matching_Explanation.ipynb

**Purpose / Propósito:**
- Demonstrates phase matching optimization
- Demonstra otimização de casamento de fase
- Shows connection to holography pipeline
- Mostra conexão com pipeline de holografia

**Topics Covered / Tópicos Cobertos:**
- Creating/loading target phase profiles / Criação/carregamento de perfis de fase alvo
- Performing phase matching optimization / Realização de otimização de casamento de fase
- Analyzing error maps / Análise de mapas de erro
- Exporting layouts for fabrication / Exportação de layouts para fabricação
- Integration with hologram design / Integração com design de holograma

**Prerequisites / Pré-requisitos:**
- Cleaned library with phase_TE, phase_TM columns
- Biblioteca limpa com colunas phase_TE, phase_TM
- Target phase maps (can be synthetic for demonstration)
- Mapas de fase alvo (podem ser sintéticos para demonstração)

**Estimated Runtime / Tempo Estimado:**
- 3-10 minutes depending on target size and library size
- 3-10 minutos dependendo do tamanho do alvo e da biblioteca

## Quick Start / Início Rápido

### Setup / Configuração

```bash
# 1. Install Jupyter (if not already installed)
pip install jupyter notebook

# 2. Navigate to notebooks directory
cd notebooks/meta_library

# 3. Start Jupyter
jupyter notebook
```

### Running Notebooks / Executando Notebooks

1. Open notebook in Jupyter interface
   Abra notebook na interface Jupyter

2. Update file paths in configuration cells
   Atualize caminhos de arquivos nas células de configuração

3. Run cells sequentially (Shift+Enter)
   Execute células sequencialmente (Shift+Enter)

4. Check output directory for generated files
   Verifique diretório de saída para arquivos gerados

## Directory Structure / Estrutura de Diretórios

```
notebooks/meta_library/
├── README.md (this file / este arquivo)
├── 01_Library_Heatmaps_Explanation.ipynb
└── 02_Phase_Matching_Explanation.ipynb
```

**Generated outputs / Saídas geradas:**
```
results/meta_library/
├── heatmaps/
│   └── notebook_example/
│       ├── heatmap_*.png
│       └── heatmap_*.npy
└── phase_matching/
    └── notebook_example/
        ├── layout_lx.csv
        ├── layout_ly.csv
        ├── layout_error_map.csv
        └── *.png
```

## Language / Idioma

All notebooks are **bilingual (English/Portuguese)** with:
- Section headers in both languages
- Code comments in English
- Markdown cells with parallel translations

Todos os notebooks são **bilíngues (Inglês/Português)** com:
- Cabeçalhos de seção em ambos os idiomas
- Comentários de código em inglês
- Células markdown com traduções paralelas

## Tips for Use / Dicas de Uso

1. **Start with small data** / **Comece com dados pequenos**
   - Use subsets for faster testing
   - Use subconjuntos para testes mais rápidos

2. **Adjust parameters** / **Ajuste parâmetros**
   - Experiment with grid sizes and colormaps
   - Experimente com tamanhos de grade e mapas de cores

3. **Save intermediate results** / **Salve resultados intermediários**
   - Export arrays for reuse in other notebooks
   - Exporte arrays para reutilização em outros notebooks

4. **Check paths** / **Verifique caminhos**
   - Update file paths to match your data location
   - Atualize caminhos de arquivos para corresponder à localização dos seus dados

## Integration with CLI Tools / Integração com Ferramentas CLI

The notebooks complement the CLI tools:
- Notebooks: **Interactive exploration and learning**
- CLI: **Batch processing and automation**

Os notebooks complementam as ferramentas CLI:
- Notebooks: **Exploração interativa e aprendizado**
- CLI: **Processamento em lote e automação**

Each notebook includes reproducibility sections showing equivalent CLI commands.

Cada notebook inclui seções de reprodutibilidade mostrando comandos CLI equivalentes.

## Further Reading / Leitura Adicional

- Module documentation: `src/meta_library/README.md`
- CLI documentation: `src/cli/README.md`
- Legacy reference code: `legacy/phase_matching/README.md`

---

## Troubleshooting / Solução de Problemas

**Problem: Module not found**
```python
# Solution: Check that src is in path
import sys
from pathlib import Path
repo_root = Path.cwd().parent.parent
sys.path.insert(0, str(repo_root / 'src'))
```

**Problem: File not found / Problema: Arquivo não encontrado**
```python
# Solution: Use absolute paths
from pathlib import Path
library_file = Path('/absolute/path/to/library.csv')
```

**Problem: Out of memory / Problema: Sem memória**
```python
# Solution: Reduce grid size or subsample data
heatmaps = phase_matching.compute_heatmaps(
    df.sample(frac=0.1),  # Use 10% of data
    bins_x=50, bins_y=50  # Reduce resolution
)
```

## Contributing / Contribuindo

To add new notebooks:
1. Follow the bilingual structure
2. Include Summary, Inputs, Run, Results, Reproducibility sections
3. Test with small datasets
4. Update this README

Para adicionar novos notebooks:
1. Siga a estrutura bilíngue
2. Inclua seções Resumo, Entradas, Execução, Resultados, Reprodutibilidade
3. Teste com conjuntos de dados pequenos
4. Atualize este README
