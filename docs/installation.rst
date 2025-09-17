Installation
============
We recommend using a virtual environment to keep your dependencies isolated. We strongly suggest installing pyCO2 in a 'Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>'_.

Step 1: Installing Anaconda
---------------------------
First, install 'Anaconda <https://www.anaconda.com/download/success>'_ for your OS.

`Thermobar <https://www.jvolcanica.org/ojs/index.php/volcanica/article/view/161>`_

Step 2: Installing PyCO2stats
-----------------------------
To create a new environment in the anaconda (or command prompt) write and run, row by row:

.. code-block:: bash

  conda create -n pyco2env python=3.10
  conda activate pyco2env

Now, there are two recommended ways to install *pyco2stats*: via the Python Package Index (PyPI) or directly from the source repository on GitHub.

Step 2.1: Installing from PyPi
------------------------------
The easiest way to install the latest stable release of pyco2stats is the following. In the anaconda or command prompt write and run:

.. code-block:: bash

  pip install pyco2stats

This will also install all required dependencies (NumPy, SciPy, pandas, matplotlib, plotly, etc.).

Step 2.2: Installing from source
--------------------------------

If you want the latest development version, in the anaconda or command prompt write and run:

.. code-block:: bash

  pip install git+https://github.com/AIVolcanoLab/pyco2stats.git

Alternatively, clone the repository manually and install in editable mode. In the anaconda or command prompt write and run, row by row:

.. code-block:: bash

  git clone https://github.com/AIVolcanoLab/pyco2stats.git
  cd pyco2stats
  pip install -e .

This setup allows you to track changes to the source code without reinstalling.

Step 4: Testing
---------------

After installation, open a Python session and run:

.. code-block:: bash

  import pyco2stats
  print(pyco2stats.__version__)

If no error appears and the version is printed, the installation was successful.

Troubleshooting
---------------

Upgrade pip, setuptools, and wheel to avoid most build issues:

.. code-block:: bash

  pip install --upgrade pip setuptools wheel
