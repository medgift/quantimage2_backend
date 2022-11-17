.. QuantImage v2 documentation master file, created by
   sphinx-quickstart on Wed Feb 26 14:54:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the QuantImage v2 Backend documentation!
===================================================

The QuantImage v2 Backend is responsible for:

1. Communicating with the `Kheops <https://kheops.online>`_ platform (fetching albums,
   downloading studies, etc.)
2. Storing data related to Kheops albums, feature extraction presets, feature extractions,
   radiomics features, patient outcomes, machine learning models, etc. in a database
3. Running feature extraction & model training tasks using `Celery <https://docs.celeryq.dev/en/stable>`_
   workers.
4. Providing a REST API to interact with all the features mentioned above to the
   `QuantImage v2 Frontend <https://github.com/medgift/quantimage2-frontend>`_.

.. note:: Research and development of QuantImage v2 was supported by the `SNSF <https://snf.ch>`_,
   `SPHN <https://sphn.ch>`_ and the `Hasler Foundation <https://haslerstiftung.ch>`_.

.. warning:: This is a research tool, not intended for clinical use.

Structure
^^^^^^^^^

Below is an overview of the structure of the QuantImage v2 Backend, showing the various Docker containers
and data flows:

.. image:: _static/backend-structure.png

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   license
   help

* :ref:`genindex`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
