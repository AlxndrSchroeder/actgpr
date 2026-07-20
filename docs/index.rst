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

.. code-block:: python

   from actgpr.objective_fn import ObjectiveFn
   from actgpr.run import OptimisationRun
   from actgpr.surrogate import GPyTorchSurrogate

   run = OptimisationRun.with_training(
       objective=ObjectiveFn(lambda x: (x - 1) ** 2),
       surrogate=GPyTorchSurrogate(),
       search_bounds=(-3.0, 5.0),
       initial_train_x=[-2.0, 4.0],
       max_evaluations=20,
       ei_threshold=0.001,
       run_dir="results",
   )
   result = run.run()
   print(result["best_x"], result["best_y"])

API reference
-------------

.. toctree::
   :maxdepth: 2

   api/actgpr

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
