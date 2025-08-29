.. pyco2stats documentation master file, created by
   sphinx-quickstart on Thu Jul  8 12:23:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyco2stats's documentation!
======================================

Introduction
============

PyCO2stats is an open-source Python library designed to perform statistical analysis and modelling of CO2 fluxes and geochemical data analysis.

PyCO2stats integrates classical and modern statistical techniques, including descriptive statystics (i.e. mean, standard deviation) and more complex analysis, such as the Gaussian Mixture Models (GMM), the Sinclair method, robust estimator for log-normal data and the Monte Carlo-based uncertainty propagation. The free library meets the transparency and reproducibility requirements and is user-friendly, being suited to be used by both Python expert and novice users.

In detail, pyco2stats is composed of 6 main classes:
   * **GMM**: this class allows to perform Gaussian Mixture Models (GMM). It enables the generation of synthetic datasets and the application of different approaches for the analysis of data with GMM;
   * **Sinclair**: this class allows to perform the graphical procedure, known as Sinclair, to partition datasets of polymodal values into two (or more) lognormal subpopulations;
   * **Propagate_Errors**: this class permits the computation of error propagation to quantify the uncertainty of fitted GMM by Monte Carlo error propagation;
   * **Stats**: collection of several statistical methods to analyze the data under investigation, with particular focus on lognormal distribution-related statistics;
   * **Viualize_MPL** and **Visualize_Plotly**: classes to perform statistical visualization either by matplotlib (Visualize_MPL) or by plotly (Visualize_Plotly).

PyCO2stats is actively maintained and improved. We value user input for feature requests and bug
reports. To contribute, you can either submit a request or report an issue directly on the GitHub Issues 
page, or reach out via email at maurizio.petrelli@unipg.it

Are you interested in developing, expanding, or suggesting changes in pyCO2stats?
To contribute, you can either `submit a request`_ or `report an issue`_ directly on the `GitHub`_,
by using the dedicated `pull request`_ and `issues`_ spaces.

.. _submit a request: https://github.com/AIVolcanoLab/pyco2stats/pulls
.. _report an issue: https://github.com/AIVolcanoLab/pyco2stats/issues
.. _pull request: https://github.com/AIVolcanoLab/pyco2stats/pulls
.. _issues: https://github.com/AIVolcanoLab/pyco2stats/issues
.. _GitHub: https://github.com/AIVolcanoLab/pyco2stats

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   modules
   installation
   citation

.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   notebooks/GMM (Gaussian Mixture Models).ipynb
   notebooks/interactive_plot_sinclaire_rev_3.ipynb
   notebooks/Sinclair.ipynb
   notebooks/Stats.ipynb
   notebooks/Visualization.ipynb

Installation
============

You can install it via pip:

.. code-block:: bash

   pip install pyco2stats

Or from source:

.. code-block:: bash

   pip install git+https://github.com/MarcoBaroni/pyco2stats.git

Citation
========
PyCO2stats is published in XXX:

**Link to the paper here**

Please cite the paper (DOI: XXX) if you are applying pyCO2stats for your study.

**References:**

Petrelli, M., Ariano, A., Ágreda-López, M., Baroni, M., Ricci, L., Frondini, F., Saldi, G., Cardellini, C., Perugini, D. and Chiodini, G. : "PyCO2stats: A Python Library for Statistical Modeling of CO2 Fluxes and
Geochemical Population Analysis in Volcanic and Environmental Systems"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
