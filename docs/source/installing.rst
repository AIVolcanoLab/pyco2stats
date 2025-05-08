Installing and Running
----------------------

We strongly suggest installing Orange-Volcanoes in a `Conda environment`_.

.. _Conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html


Step 1: Installing Anaconda or Miniconda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, install `Anaconda`_ or `Miniconda`_ for your OS.  
Should you use Anaconda Distribution or Miniconda? If you are unsure, please refer to the
`Getting Started with Anaconda guide`_.

.. _Anaconda: https://www.anaconda.com/download/success
.. _Miniconda: https://www.anaconda.com/download/success#miniconda
.. _Getting Started with Anaconda guide: https://docs.anaconda.com/getting-started/


Step 2: Installing pyco2stats
~~~~~~~~~~~~~~~~~~~~~~~~~

Then, create a new conda environment, and install `pyco2stats`_:

.. _pyco2stats: https://github.com/AIVolcanoLab/pyco2stats/

.. code-block:: shell

   # Add conda-forge to your channels for access to the latest release
   conda config --add channels conda-forge

   # Perhaps enforce strict conda-forge priority
   conda config --set channel_priority strict

   # Create and activate an environment for Orange
   conda create python=3.10 --yes --name co2stats
   conda activate co2stats

   # Install Orange
   conda install pyco2stats
