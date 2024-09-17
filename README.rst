#########
fouriever
#########

A single toolkit for calibrations, correlations, and companion search in kernel phase, aperture masking, and long-baseline interferometry data. Supported data formats are :code:`OIFITS` and :code:`KPFITS`.

Installation
************

At this stage, it is recommended to clone the Git repository for installation:

::

	git clone https://github.com/kammerje/fouriever.git

From here, it is highly recommended that you create a unique Conda environment to hold all of the fouriever dependencies:

::

	conda create --name fouriever python=3.11
	conda activate fouriever

With the Conda environment created and activated, move to the cloned directory and install the dependencies and fouriever itself:

::

	cd where/you/saved/the/git/repo
	pip install -e .
