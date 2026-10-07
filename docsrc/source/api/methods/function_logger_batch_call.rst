=============================
``FunctionLogger.batch_call``
=============================

.. automethod:: pyvbmc.function_logger.FunctionLogger.batch_call

For the logger owned by a ``VBMC`` instance, the logged function is VBMC's
assembled log joint, so values passed directly to ``batch_call`` include any
separately supplied prior. This differs from
``VBMC(precomputed_evaluations=(X, y))``: there, ``y`` contains outputs of the
``log_density`` argument, and VBMC adds a separately supplied prior once. See
:doc:`the VBMC interface <../classes/vbmc>` for that higher-level workflow.
