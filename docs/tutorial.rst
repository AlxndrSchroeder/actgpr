Tutorial
========

This tutorial walks through a complete optimisation run: wrapping a blackbox
function in an Objective, configuring the search, executing the run, and
reading the reproducibility record it leaves behind.

Setup
-----

.. code-block:: bash

   git clone https://github.com/LxdrScr/actgpr.git
   cd actgpr
   poetry install

Step 1 — wrap your blackbox function
------------------------------------

``actgpr`` minimises a scalar blackbox function: something that takes one
float and returns one float. In practice that might launch a simulation or
trigger an experiment — here we use an analytic stand-in so the tutorial
runs instantly:

.. code-block:: python

   from actgpr import ObjectiveFn

   def my_blackbox(x: float) -> float:
       """Stand-in for a simulation or experiment."""
       return (x - 1) ** 2

   objective = ObjectiveFn(my_blackbox)

``ObjectiveFn`` turns any ``Callable[[float], float]`` into an Objective.
Its ``evaluate`` method accepts one or more input points and returns a tuple
of outputs:

.. code-block:: python

   objective.evaluate(0.0)        # (1.0,)
   objective.evaluate(0.0, 3.0)   # (1.0, 4.0)

Errors raised inside your function propagate unchanged, so you can handle
them by their original type.

Step 2 — configure the run
--------------------------

Three decisions matter most:

``search_bounds``
    The closed interval ``[lo, hi]`` in which the algorithm searches for the
    minimum. The blackbox is never evaluated outside it.

``max_evaluations``
    The budget cap: the maximum number of active optimisation iterations
    (GPR fit cycles).

``ei_threshold``
    The convergence threshold: the run stops early once the best achievable
    Expected Improvement falls below this value — meaning the surrogate sees
    nothing left to gain.

You also choose a **fit mode**:

- ``OptimisationRun.with_training(...)`` — the GP hyperparameters
  (lengthscale, outputscale, noise) are optimised at every iteration.
  Use this when you do not know good hyperparameters — the usual case.
- ``OptimisationRun.without_training(...)`` — hyperparameters stay fixed at
  the values you pass. Use this for controlled comparisons or when good
  values are already known.

.. code-block:: python

   from actgpr import GPyTorchSurrogate, OptimisationRun

   run = OptimisationRun.with_training(
       objective=objective,
       surrogate=GPyTorchSurrogate(),
       search_bounds=(-3.0, 5.0),   # interval in which the minimum is searched
       initial_train_x=[-2.0, 4.0],  # seed points to fit the first surrogate
       max_evaluations=20,
       ei_threshold=0.001,
       store_snapshots=True,         # keep per-iteration state for plotting
       run_dir="results",            # write the MRR record
   )

Step 3 — execute and interpret
------------------------------

.. code-block:: python

   result = run.run()

   print(result["best_x"])       # ≈ 1.0  — input point with the lowest output
   print(result["best_y"])       # ≈ 0.0  — the lowest output found
   print(result["n_iterations"])  # iterations actually executed
   print(result["stop_reason"])   # "ei_threshold" or "max_evaluations"

``result["train_x"]`` and ``result["train_y"]`` hold every input point the
run evaluated and the corresponding Objective outputs — the initial points
first, then one point per iteration.

Step 4 — browse the iterations
------------------------------

Because the run was created with ``store_snapshots=True``, you can step
through the surrogate's view of the problem iteration by iteration:

.. code-block:: python

   run.plot_iterations()

An interactive matplotlib window opens with the GP prediction (mean, 95 %
confidence band, training data) on top, the EI landscape below, and a
slider to scrub through iterations.

Step 5 — the reproducibility record (MRR)
-----------------------------------------

Because ``run_dir`` was given, the run created a timestamped folder under
``results/`` with five artifacts: ``config.json`` (parameters),
``manifest.json`` (input checksums), ``meta.json`` (environment and output
summary), ``run.log`` (per-iteration audit trail), and ``results.h5``.

``results.h5`` is self-describing — the configuration is stored as HDF5
attributes next to the data. The per-iteration history reads directly as
plottable series:

.. code-block:: python

   from pathlib import Path

   import h5py
   import matplotlib.pyplot as plt

   run_dir = sorted(Path("results").iterdir())[-1]   # newest run

   with h5py.File(run_dir / "results.h5") as f:
       iteration = f["history/iteration"][:]
       pred_error = f["history/prediction_error"][:]
       improvement = f["history/improvement"][:]

   plt.plot(iteration, pred_error, label="prediction_error")
   plt.plot(iteration, improvement, label="improvement")
   plt.xlabel("iteration")
   plt.legend()
   plt.show()

``prediction_error`` (surrogate error at the chosen point) shrinking towards
zero tells you the surrogate is learning the blackbox; ``improvement``
flattening tells you the optimisation has converged.

Where to go next
----------------

- The :doc:`API reference <api/actgpr>` documents every class and function.
- The README's vocabulary section defines every term used in this package.
