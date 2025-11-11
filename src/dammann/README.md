# Dammann FFT - Grades de Dammann para Spot Clouds

Script principal para geração de **grades de Dammann** usando **algoritmo GS + FFT**. Desenvolvido para criar metassuperfícies periódicas que produzem padrões uniformes de spots no far-field.

## 🎯 O Que Faz

Este script calcula supercélulas fase-únicas que, quando replicadas em mosaico, geram **spot clouds** uniformes no plano de Fourier. É a ferramenta que usamos para projetar elementos difrativos com múltiplos feixes.

**Funcionalidades principais:**

- Calcula mapas de fase para supercélulas usando GS
- Replica em mosaico para criar metassuperfícies completas
- Analisa eficiência difrativa e uniformidade dos spots
- Gera visualizações completas do padrão de difração

## 🚀 Uso Rápido

### Linha de Comando (CLI)

```bash
# Exemplo básico - usa parâmetros padrão
python src/holography/dammann_fft.py --experiment teste_dammann

# Exemplo completo com todos os parâmetros
python src/dammann/dammann_fft.py \
  --wavelength 1.064e-6 \
  --P 5.2e-7 \
  --supercell_pixels 45 \
  --n_supercells 10 \
  --iters 400 \
  --seed 0 \
  --experiment meu_dammann \
  --pol Y
```

### Como Módulo Python

```python
from src.dammann.dammann_fft import run_dammann_batch
from pathlib import Path

resultado = run_dammann_batch(
    out_root=Path("results/holography/dammann"),
    experiment="teste_programatico",
    pol_label="Y"
)
```

## 📁 Estrutura de Saída

Cada execução cria uma pasta organizada com timestamp:

```
results/holography-dammann/dammann/meu_experimento/2024-01-15_14-30-00/
├── phase_map__meu_experimento__Y__λ_1064nm__P_520nm__scpix_45px__nsc_10__iter_400__seed_0.txt
├── phase_map__meu_experimento__Y__λ_1064nm__P_520nm__scpix_45px__nsc_10__iter_400__seed_0.png
├── convergence__meu_experimento__Y__λ_1064nm__P_520nm__scpix_45px__nsc_10__iter_400__seed_0.txt
├── convergence__meu_experimento__Y__λ_1064nm__P_520nm__scpix_45px__nsc_10__iter_400__seed_0.png
├── diffraction_orders__meu_experimento__Y__λ_1064nm__P_520nm__scpix_45px__nsc_10__iter_400__seed_0.png
├── diffraction_orders__meu_experimento__Y__λ_1064nm__P_520nm__scpix_45px__nsc_10__iter_400__seed_0_table.csv
├── summary__meu_experimento__Y__λ_1064nm__P_520nm__scpix_45px__nsc_10__iter_400__seed_0.png
└── run_meta.json
```

## ⚙️ Parâmetros Principais

| Parâmetro             | Default      | Descrição                              |
| ---------------------- | ------------ | ---------------------------------------- |
| `--wavelength`       | 1064e-9      | Comprimento de onda [m]                  |
| `--P`                | 520e-9       | Tamanho do pixel [m]                     |
| `--supercell_pixels` | 45           | Pixels por lado da supercélula          |
| `--n_supercells`     | 10           | Número de supercélulas por lado        |
| `--iters`            | 400          | Iterações do algoritmo GS              |
| `--seed`             | 0            | Semente para reprodutibilidade           |
| `--experiment`       | demo_dammann | Nome do experimento                      |
| `--pol`              | Y            | Polarização (X/Y) - para organização |

## 📊 Métricas Calculadas

- **DE (Diffraction Efficiency)**: Fração de energia nas ordens propagantes
- **RMSE (Uniformidade)**: Erro quadrático médio da uniformidade dos spots
- **M_orders**: Número de ordens de difração propagantes

## 🧩 No Contexto do Repositório

Este módulo faz parte de um conjunto de ferramentas para holografia:

- **`gs_asm.py`** - GS com Espectro Angular para hologramas de imagem
- **`dammann_fft.py`** (este) - Grades de Dammann para spot clouds ← *Você está aqui*
- **`meta_library.py`** - Utilitários comuns para metassuperfícies

Baseado nas células do notebook:

- `notebooks/holography/explanations/02_GS_PolarizationY_Explanation.ipynb`

## 💡 Dicas Práticas

### Controle de Reprodutibilidade

Use `--seed` para garantir resultados idênticos entre execuções. Útil para debug e comparações.

### Tamanho da Supercélula

- `supercell_pixels` controla a resolução do padrão de fase
- Valores maiores permitem padrões mais complexos mas aumentam o tempo de cálculo

### Número de Supercélulas

- `n_supercells` define o tamanho final da metassuperfície
- Afeta a resolução do padrão de difração no far-field

### Otimização de Parâmetros

Para melhor uniformidade, aumente `--iters`. Típico: 200-500 iterações.

## 🔍 Para Saber Mais

- O algoritmo GS é aplicado apenas na supercélula, não na metassuperfície completa
- O far-field é calculado via FFT da metassuperfície completa
- As ordens de difração são amostradas nos pontos (p/λ, q/λ) do espaço k
- A eficiência considera apenas ordens dentro do cone de propagação

---

*Parte do toolkit de holografia do repositório - [Voltar ao README principal](../../README.md)*
