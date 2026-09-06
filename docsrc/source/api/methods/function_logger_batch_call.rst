=============================
``FunctionLogger.batch_call``
=============================

.. automethod:: pyvbmc.function_logger.FunctionLogger.batch_call

``x`` is a nonempty ``(N, D)`` array in transformed coordinates. The method
is available when the logger uses a vectorized target. That target receives
all missing rows in one ``(M, D)`` original-coordinate array and returns
values with shape ``(M,)`` or ``(M, 1)``.

Pass ``f_vals`` to reuse initial log-joint values. It has length ``N`` and a
``NaN`` marks each row that must be evaluated. Cached values already include
the prior. The returned values, optional noise standard deviations, and cache
indices all follow the row order of ``x``, including when cached and evaluated
rows are interleaved.

For user-provided target noise, the target returns ``(values, sds)`` as two
arrays of shape ``(M,)`` or ``(M, 1)``. An ``(M, 2)`` array is not accepted.
