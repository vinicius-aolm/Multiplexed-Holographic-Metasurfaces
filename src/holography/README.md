## GS+ASM - Geração de Hologramas de Fase

Script principal para cálculo de hologramas usando **Gerchberg-Saxton** com **Espectro Angular**. Desenvolvido para nossos experimentos.

## 🎯 O Que Faz

Este script pega imagens comuns (como logos da UFABC ou ILUM) e calcula os padrões de fase necessários para reproduzi-las como hologramas. É a implementação que usamos rotineiramente no laboratório.

**Funcionalidades principais:**

- Calcula mapas de fase a partir de imagens alvo
- Simula a propagação óptica usando método do espectro angular
- Gera figuras de análise e validação automáticas
- Organiza resultados de forma estruturada e reproduzível

## 🚀 Uso Rápido

### Linha de Comando (CLI)

```bash
# Exemplo básico - usa parâmetros padrão do nosso setup
python src/holography/gs_asm.py --targets ilum.png ufabc.png --experiment teste_rapido

# Exemplo completo com todos os parâmetros
python src/holography/gs_asm.py \
  --wavelength 1.064e-6 \
  --z 3.8e-4 \
  --dx 5.2e-7 \
  --NA 0.65 \
  --iters 200 \
  --targets ilum.png ufabc.png \
  --experiment meu_experimento \
  --pol X
```

### Como Módulo Python

```python
from src.holography.gs_asm import run_batch
from pathlib import Path

# Configuração básica
targets = [
    ("ilum", Path("data/targets/common/ilum.png")),
    ("ufabc", Path("data/targets/common/ufabc.png")),
]

resultado = run_batch(
    targets=targets,
    out_root=Path("results/holography/gs_x"),
    experiment="teste_programatico",
    pol_label="X"
)
```

## 📁 Estrutura de Saída

Cada execução cria uma pasta organizada com timestamp:

```
results/holography/gs_x/meu_experimento/2024-01-15_14-30-00/
├── ilum/                                  # Pasta do primeiro alvo
│   ├── phase_map__ilum__X__λ_1064nm__z_380um__dx_520nm__iter_200.txt
│   ├── imagem_alvo__ilum__X__λ_1064nm__z_380um__dx_520nm__iter_200.png
│   ├── mapa_de_fase__ilum__X__λ_1064nm__z_380um__dx_520nm__iter_200.png
│   ├── reconstruida__ilum__X__λ_1064nm__z_380um__dx_520nm__iter_200.png
│   ├── convergencia__ilum__X__λ_1064nm__z_380um__dx_520nm__iter_200.png
│   └── sumario_alvo__ilum__X__λ_1064nm__z_380um__dx_520nm__iter_200.png
├── ufabc/                                 # Pasta do segundo alvo
│   └── ... (mesma estrutura)
├── summary_meu_experimento__ilum__ufabc__X__λ_1064nm__z_380um__dx_520nm__iter_200.png
└── run_meta.json                         # Metadados da execução
```

## ⚙️ Parâmetros Principais

| Parâmetro       | Default             | Descrição                              |
| ---------------- | ------------------- | ---------------------------------------- |
| `--wavelength` | 1064e-9             | Comprimento de onda [m]                  |
| `--z`          | 380e-6              | Distância de propagação [m]           |
| `--dx`         | 520e-9              | Tamanho do pixel no SLM [m]              |
| `--NA`         | 0.65                | Abertura numérica                       |
| `--iters`      | 200                 | Iterações do algoritmo GS              |
| `--targets`    | ilum.png, ufabc.png | Imagens a processar                      |
| `--experiment` | demo_holografia     | Nome do experimento                      |
| `--pol`        | X                   | Polarização (X/Y) - para organização |

## 🧩 No Contexto do Repositório

Este módulo faz parte de um conjunto de ferramentas para holografia:

- **`gs_asm.py`** (este) - GS com Espectro Angular ← *Você está aqui*
- **`damman_fft.py`** - Grades de Dammann via FFT
- **`meta_library.py`** - Utilitários comuns para metassuperfícies

Para teoria detalhada, veja os notebooks explicativos:

- `notebooks/holography/explanations/01_GS_PolarizationX_Explanation.ipynb`
- `notebooks/holography/explanations/02_GS_PolarizationY_Explanation.ipynb`

## 💡 Dicas Práticas

### Fallback de Imagens

Se uma imagem não for encontrada, o script cria automaticamente um padrão de teste em formato de "H". Útil para testes rápidos.

### Limite de Comparação

Quando processa mais de 2 imagens, **não gera** o sumário comparativo para evitar figuras muito grandes. Para comparações, processe no máximo 2 por vez.

### Organização por Polarização

O parâmetro `--pol` não afeta a física, só a organização. Use para separar execuções de diferentes configurações experimentais.

### Windows vs Linux

- **Windows (PowerShell):** Use `` ` `` para quebras de linha
- **Linux/macOS:** Use `\` para quebras de linha

## 🔍 Para Saber Mais

- Os parâmetros padrão refletem nosso setup atual
- A correlação de Pearson é usada como métrica de convergência
- O filtro de NA remove componentes evanescentes não propagantes
- Todos os resultados incluem metadados completos para reprodução

---

*Parte do toolkit de holografia do repositório - [Voltar ao README principal](../../README.md)*
