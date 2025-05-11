__author__ = 'Maurizio Petrelli, Alessandra Ariano, Baroni Marco, Frondini Francesco, Ágreda-López Mónica, Chiodini Giovanni'

# PyCO2stats/__init__.py
from .gaussian_mixtures import GMM
from .sinclair import Sinclair
from .stats import Stats
from .visualize_mpl import Visualize_Mpl
from .propagate_errors import Propagate_Errors
from .visualize_plotly import Visualize_Plotly
#from .env_stats_py import EnvStatsPy

#__all__ = ["DataHandler", "GMM", "Visualize_Mpl", "Sinclair", "Stats", "Propagate_Errors", "Visualize_Plotly"]#"EnvStatsPy", "Visualize_Plotly"]
__all__ = ["GMM", "Sinclair", "Stats", "Propagate_Errors", "Visualize_Mpl", "Visualize_Plotly"]
