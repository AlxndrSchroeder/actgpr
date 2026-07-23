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

``max_iterations``
    The budget cap: the maximum number of active optimisation iterations
    (GPR fit cycles).

``ei_threshold``
    The convergence threshold: the run stops early once the best achievable
    Expected Improvement falls below this value — meaning the surrogate sees
    nothing left to gain.

You also choose a **fit mode**:

- ``OptimisationRun.with_training(...)`` — the GP hyperparameters
  (lengthscale, outputscale, noise) are re-tuned at every iteration using
  `Adam <https://arxiv.org/abs/1412.6980>`_ (GPyTorch's ``torch.optim.Adam``
  integration), a gradient-descent variant that maximises the marginal log
  likelihood — how plausible the observed training data is under the GP.
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
       initial_train_x=[-3.0, 5.0],  # points where we start looking for the minimum
       max_iterations=20,
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
   print(result["stop_reason"])   # "ei_threshold" or "max_iterations"

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
``results/`` holding the five MRR artifacts:

- ``config.json`` — every parameter used, written at the start of the run
- ``manifest.json`` — a SHA-256 checksum of the inputs
- ``meta.json`` — environment: package name, version, and repository, git
  commit, Python/library versions, platform, timestamps, and output summary
- ``run.log`` — a human-readable, per-iteration audit trail
- ``results.h5`` — self-describing HDF5: configuration is stored as
  attributes alongside the data, so the file can be understood on its own

Revisiting a saved run
~~~~~~~~~~~~~~~~~~~~~~~

``plot_run_history`` builds the validation-metrics plot directly from a
run directory — no ``OptimisationRun`` object needed, so a past run can be
revisited at any later time:

.. code-block:: python

   from pathlib import Path

   from actgpr.plotting import plot_run_history

   run_dir = sorted(Path("results").iterdir())[-1]   # newest run
   plot_run_history(run_dir)

This plots ``prediction_error`` and ``improvement`` against iteration:
``prediction_error`` shrinking towards zero shows the surrogate learning
the blackbox; ``improvement`` flattening shows the optimisation converging.

For a custom analysis, read the same series directly:

.. code-block:: python

   import h5py

   with h5py.File(run_dir / "results.h5") as f:
       iteration = f["history/iteration"][:]
       prediction_error = f["history/prediction_error"][:]
       improvement = f["history/improvement"][:]

Parameter reference
--------------------

``with_training`` and ``without_training`` share the same core parameters
and differ only in how the GP hyperparameters are handled.

Shared parameters
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Meaning
   * - ``objective``
     - The wrapped blackbox function to minimise (an ``ObjectiveFn``).
   * - ``surrogate``
     - The GP surrogate backend. Pass a fresh ``GPyTorchSurrogate()`` — it
       holds fitted state internally, so do not reuse one across runs.
   * - ``search_bounds``
     - The closed interval ``(lo, hi)`` in which the algorithm searches for
       the minimum. The blackbox is never evaluated outside it.
   * - ``initial_train_x``
     - Points where the search starts — the first surrogate is fitted to
       these before the loop runs. By convention, use the two
       ``search_bounds`` endpoints.
   * - ``max_iterations``
     - Budget cap: the maximum number of active optimisation iterations
       (GPR fit cycles) — not individual blackbox evaluations.
   * - ``ei_threshold``
     - Convergence threshold: the run stops early once the best achievable
       Expected Improvement drops below this value.
   * - ``n_candidates`` (default 500)
     - Number of evenly spaced candidate points the acquisition function
       scores every iteration.
   * - ``noise`` (default 1e-4)
     - Starting observation noise variance for the GP likelihood. In
       ``with_training`` it is only a *starting point* — Adam tunes it
       further alongside lengthscale and outputscale. In
       ``without_training`` it stays fixed at this value for the whole run.
   * - ``store_snapshots`` (default False)
     - If ``True``, also keeps each iteration's full GP/EI arrays (in
       memory and under ``results.h5``'s ``iterations/`` group) so
       ``plot_iterations()`` can browse them afterward. The
       ``prediction_error``/``improvement`` history used by
       ``plot_run_history()`` is recorded either way.
   * - ``run_dir`` (default None)
     - If given, writes the MRR record (see Step 5) to a timestamped
       folder under this path. If ``None``, nothing is written to disk.

``with_training`` only
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Meaning
   * - ``training_iter`` (default 50)
     - Number of Adam optimisation steps run per surrogate fit, tuning
       lengthscale, outputscale, and noise to maximise the marginal log
       likelihood.

``without_training`` only
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Meaning
   * - ``lengthscale`` (default 1.0)
     - Fixed RBF kernel lengthscale — never tuned.
   * - ``outputscale`` (default 1.0)
     - Fixed kernel signal variance — never tuned.

Where to go next
----------------

- The :doc:`API reference <api/actgpr>` documents every class and function.
- The README's vocabulary section defines every term used in this package.
