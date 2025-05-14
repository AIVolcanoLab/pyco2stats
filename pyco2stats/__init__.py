__author__ = 'Maurizio Petrelli, Alessandra Ariano'


# PyCO2stats/__init__.py
from .gaussian_mixtures import GMM
from .sinclair import Sinclair
from .stats import Stats
from .visualize_mpl import Visualize_Mpl
from .propagate_errors import Propagate_Errors
from .visualize_plotly import Visualize_Plotly

__all__ = ["DataHandler", "GMM", "Visualize_Mpl", "Sinclair", "Stats", "Propagate_Errors", "Visualize_Plotly"]