Installation
============

There are two recommended ways to install **pyco2stats**:  
via the Python Package Index (PyPI) or directly from the source repository on GitHub.

Prerequisites
-------------

We recommend using a virtual environment (``venv`` or ``conda``) to keep your dependencies isolated.


To create a new environment using venv:
.. code-block:: bash

   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

While with conda:
.. code-block:: bash

  conda create -n pyco2env python=3.10
  conda activate pyco2env

Installing from PyPI
--------------------

The easiest way to install the latest stable release of pyco2stats is:

.. code-block:: bash

   pip install pyco2stats

This will also install all required dependencies (NumPy, SciPy, pandas, matplotlib, plotly, etc.).

Installing from Source
----------------------

If you want the latest development version:

.. code-block:: bash

   pip install git+https://github.com/AIVolcanoLab/pyco2stats.git

Alternatively, clone the repository manually and install in *editable* mode:

.. code-block:: bash

   git clone https://github.com/AIVolcanoLab/pyco2stats.git
   cd pyco2stats
   pip install -e .

This setup allows you to track changes to the source code without reinstalling.

Testing the Installation
------------------------

After installation, open a Python session and run:

.. code-block:: python

   import pyco2stats
   print(pyco2stats.__version__)

If no error appears and the version is printed, the installation was successful.

Troubleshooting
---------------

Upgrade **pip**, **setuptools**, and **wheel** to avoid most build issues:

.. code-block:: bash

  pip install --upgrade pip setuptools wheel
