actgpr — Active GPR Optimisation
================================

``actgpr`` finds the minimum of an expensive-to-evaluate scalar Objective by
iteratively fitting a Gaussian Process Surrogate and using an Acquisition
function (Expected Improvement) to pick the most informative next input point.

Every run can produce a Minimal Reproducible Run (MRR) record: ``config.json``,
``manifest.json``, ``meta.json``, ``run.log``, and a self-describing
``results.h5``.

Quick example
-------------

Wrap your blackbox function in an ``ObjectiveFn``, choose the search
interval via ``search_bounds``, and hand both to an ``OptimisationRun``:

.. code-block:: python

   from actgpr import ObjectiveFn, OptimisationRun, GPyTorchSurrogate

   def my_blackbox(x: float) -> float:
       return (x - 1) ** 2   # stand-in for a simulation or experiment

   run = OptimisationRun.with_training(
       objective=ObjectiveFn(my_blackbox),
       surrogate=GPyTorchSurrogate(),
       search_bounds=(-3.0, 5.0),   # interval in which the minimum is searched
       initial_train_x=[-2.0, 4.0],
       max_evaluations=20,
       ei_threshold=0.001,
       run_dir="results",
   )
   result = run.run()
   print(result["best_x"], result["best_y"])

Documentation
-------------

.. toctree::
   :maxdepth: 2

   tutorial
   api/actgpr

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
