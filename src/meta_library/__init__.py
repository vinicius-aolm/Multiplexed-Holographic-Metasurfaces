"""
Meta Library Module for Metasurface Data Processing.

This package provides tools for:
- Parsing Touchstone files with S-parameter data
- Computing derived transmission parameters (amplitude, phase)
- Generating heatmaps of parameter spaces
- Performing phase matching optimization

Módulo de Meta Biblioteca para Processamento de Dados de Metassuperfície.
"""

from . import generate_df
from . import clean_library
from . import phase_matching

__all__ = ['generate_df', 'clean_library', 'phase_matching']
__version__ = '1.0.0'
