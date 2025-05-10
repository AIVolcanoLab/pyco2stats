.. pyco2stats documentation master file, created by
   sphinx-quickstart on Thu Jul  8 12:23:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyco2stats's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

installing

introduction

modules

Welcome to the documentation for pyco2stats!

Introduction
============

Welcome to pyco2stats.
PyCO2stats is a free open source Python library desinged to perform both classic and robust statistical methods for CO2 flux and geochemical data analysis.

PyCO2stats bridges the gap between advanced data science techniques and the specific requirements
of geochemical datasets regarding CO2 fluxes. PyCO2stats is designed to grant the maximum flexibility possible, enabling the user to perform a vast range of analyses, from classical statistical methods (such as mean, standard deviation etc...) to more complete and complex analyses, such as Gaussian Mixture Models (GMM) or lognormal statistics for reduced population sizes. Moreover PyCO2stats meets the requirements in term for transparency and reproducibility, meeting so the FAIR (Findability, Accessibility, Interoperability and Reusability) and OpenScience standards. Laslty, PyCO2stats is suited for used either by Python expert users or by more novice users.

In detail pyco2stats is composed of 6 main classes:
   * **GMM**: this class enables working with Gaussian Mixture Models (GMM). It permits the generation of synthetic datasets and the application of different approaches for the analysis of data with GMM;
   * **Sinclair**: this class enables the possiblity to perform the graphical procedure known as Sinclair to partition datasets of polymodal values into two (or more) lognormal subpopulations;
   * **Propagate_Errors**: class that permits the computation of error propagation to quantify the uncertainety of fitted GMM by Monte Carlo error propagation;
   * **Stats**: collection of several statistical methods to analyze the data uncer investigation, with particular focus on lognormal distribution related statistics;
   * **Viualize_MPL** and **Visualize_Plotly**: classes to perform statistical visualization either by matplotlib (Visualize_MPL) or by plotly (Visualize_Plotly).

PyCO2stats is actively maintained and improved. We value user input for feature requests and bug
reports. To contribute, you can either submit a request or report an issue directly on the GitHub Issues
page, or reach out via email at XXX.

Are you interested in developing, expanding, or suggesting changes in pyCO2stats?
To contribute, you can either `submit a request`_ or `report an issue`_ directly on the `GitHub`_,
by using the dedicated `pull request`_ and `issues`_ spaces.

.. _submit a request: https://github.com/AIVolcanoLab/pyco2stats/pulls
.. _report an issue: https://github.com/AIVolcanoLab/pyco2stats/issues
.. _pull request: https://github.com/AIVolcanoLab/pyco2stats/pulls
.. _issues: https://github.com/AIVolcanoLab/pyco2stats/issues
.. _GitHub: https://github.com/AIVolcanoLab/pyco2stats


Installation
============

You can install it via pip:

.. code-block:: bash

   pip install mylibrary

Or from source:

.. code-block:: bash

   git clone https://github.com/AIVolcanoLab/pyco2stats.git
   cd mylibrary
   pip install .

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========
PyCO2stats is published in XXX:

**Link to the paper here**

Please cite the paper (DOI: XXX) if you are applying pyCO2stats for your study.

**References:**

Petrelli, M., Ariano, A., Baroni, M., Frondini, F., Ágreda-López, M. and Chiodini, G. : "PyCO2stats: A Python Library for Statistical Modeling of CO2 Fluxes and
Geochemical Population Analysis in Volcanic and Environmental Systems"
