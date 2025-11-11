# Processamento de Biblioteca de Metassuperfície

Este documento explica como usar o notebook `01_Library_Heatmaps_Explanation.ipynb` e processar novas bibliotecas.

## 📚 Usando o Notebook

O notebook `notebooks/meta_library/01_Library_Heatmaps_Explanation.ipynb` está agora completamente executável e demonstra:

1. **Importação de bibliotecas** processadas de metassuperfícies
2. **Geração de heatmaps** para visualizar cobertura de fase e amplitude
3. **Análise de cobertura** no espaço de fases
4. **Salvamento de resultados** (heatmaps e visualizações)

### Execução

```bash
cd notebooks/meta_library
jupyter notebook 01_Library_Heatmaps_Explanation.ipynb
```

Ou execute todas as células:

```bash
jupyter nbconvert --to notebook --execute 01_Library_Heatmaps_Explanation.ipynb
```

## 🔧 Processamento de Novas Bibliotecas

Se você tem novos dados brutos de bibliotecas (arquivos CSV com parâmetros S), use o script de processamento:

### Uso do Script

```bash
python scripts/process_raw_library.py <caminho_entrada.csv> [caminho_saida.csv]
```

### Exemplo

Processar a biblioteca de altura variável:

```bash
python scripts/process_raw_library.py \
    Bibliotecas/Altura_Varia/biblioteca_Bib1-27x27-perdas.csv \
    data/meta_library/library_cleaned.csv
```

### O que o script faz:

1. **Carrega** o arquivo CSV bruto
2. **Filtra** por frequência (usa a primeira se múltiplas)
3. **Calcula** propriedades ópticas:
   - `phase_TE` = arctan2(S13_imag, S13_real)
   - `amp_TE` = sqrt(S13_real² + S13_imag²)
   - `phase_TM` = arctan2(S24_imag, S24_real)
   - `amp_TM` = sqrt(S24_real² + S24_imag²)
4. **Salva** CSV processado com colunas: L_x, L_y, H, phase_TE, amp_TE, phase_TM, amp_TM

## 📊 Estrutura dos Dados

### Biblioteca Bruta (Entrada)

Arquivo CSV com parâmetros S de simulações/medições:
- `L_x`, `L_y`: Dimensões da nanoestrutura (nm)
- `H`: Altura (nm)
- `S13_real`, `S13_imag`: Coeficiente de transmissão TE
- `S24_real`, `S24_imag`: Coeficiente de transmissão TM
- `frequencia_ghz`: Frequência de operação

### Biblioteca Processada (Saída)

Arquivo CSV limpo e pronto para uso:
- `L_x`, `L_y`: Dimensões (nm)
- `H`: Altura (nm)
- `phase_TE`: Fase TE [radianos]
- `amp_TE`: Amplitude TE [0-1]
- `phase_TM`: Fase TM [radianos]
- `amp_TM`: Amplitude TM [0-1]

## 🗂️ Bibliotecas Disponíveis

Este repositório inclui várias bibliotecas na pasta `Bibliotecas/`:

- `Altura_Varia/`: Biblioteca com 196 geometrias diferentes (27x27 grid, perdas incluídas)
- `27x27_CaixaColada/`: Biblioteca 27x27 com configuração de caixa colada
- `196/`: Biblioteca compacta com 196 entradas
- `729/`: Biblioteca expandida com 729 entradas
- `RGB_10x10/`: Biblioteca para aplicações RGB
- `Colorido/`: Biblioteca para holografia colorida

## 🔄 Workflow Típico

1. **Simular/Medir** nanoestruturas → Gerar arquivos Touchstone (.ts)
2. **Converter** .ts para CSV usando `src/meta_library/generate_df.py`
3. **Processar** CSV usando `scripts/process_raw_library.py`
4. **Usar** biblioteca processada no notebook `01_Library_Heatmaps_Explanation.ipynb`
5. **Analisar** heatmaps e cobertura de fase
6. **Aplicar** em design de metassuperfícies (notebook `02_Phase_Matching_Explanation.ipynb`)

## 📖 Referências

- Notebook legado: `legacy/phase_matching/HeatmapsGenerator.ipynb`
- Módulo de processamento: `src/meta_library/clean_library.py`
- Módulo de geração: `src/meta_library/generate_df.py`
- Módulo de casamento de fase: `src/meta_library/phase_matching.py`

## ✅ Verificação

Para verificar se tudo está funcionando:

```bash
# 1. Processar uma biblioteca
python scripts/process_raw_library.py \
    Bibliotecas/Altura_Varia/biblioteca_Bib1-27x27-perdas.csv \
    data/meta_library/library_test.csv

# 2. Executar o notebook
cd notebooks/meta_library
jupyter nbconvert --to notebook --execute 01_Library_Heatmaps_Explanation.ipynb

# 3. Verificar saídas
ls -lh ../../results/meta_library/heatmaps/demo_*/
```

Se todos os passos funcionarem sem erros, o sistema está pronto!
