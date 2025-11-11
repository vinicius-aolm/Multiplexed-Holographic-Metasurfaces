"""
Módulo de Meta Biblioteca para Processamento de Dados de Metassuperfície.

Este pacote fornece ferramentas para:
- Análise de arquivos Touchstone com dados de parâmetros S
- Cálculo de parâmetros de transmissão derivados (amplitude, fase)
- Geração de mapas de calor de espaços de parâmetros
- Realização de otimização de casamento de fase

Módulos:
    generate_df: Análise de arquivos Touchstone
    clean_library: Limpeza de dados e colunas derivadas  
    phase_matching: Mapas de calor e casamento de fase
"""

from . import generate_df
from . import clean_library
from . import phase_matching

__all__ = ['generate_df', 'clean_library', 'phase_matching']
__version__ = '1.0.0'
