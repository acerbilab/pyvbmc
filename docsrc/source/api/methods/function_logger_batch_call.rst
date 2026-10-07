=============================
``FunctionLogger.batch_call``
=============================

.. automethod:: pyvbmc.function_logger.FunctionLogger.batch_call

``x`` is a nonempty ``(N, D)`` array in transformed coordinates. The method
is available when the logger uses a vectorized target. That target receives
all missing rows in one ``(M, D)`` original-coordinate array and returns
values with shape ``(M,)`` or ``(M, 1)``.

Pass ``f_vals`` to reuse values of the same logged function at the
corresponding rows. It has length ``N`` and a ``NaN`` marks each row that must
be evaluated. These values are in the original function's scale, before any
parameter-transform Jacobian is added. The returned values, optional noise
standard deviations, and cache indices all follow the row order of ``x``,
including when supplied and evaluated rows are interleaved.

For the logger owned by a ``VBMC`` instance, the logged function is VBMC's
assembled log joint, so values passed directly to ``batch_call`` include any
separately supplied prior. This differs from
``VBMC(precomputed_evaluations=(X, y))``: there, ``y`` contains outputs of the
``log_density`` argument, and VBMC adds a separately supplied prior once.

For user-provided target noise, the target returns ``(values, sds)`` as two
arrays of shape ``(M,)`` or ``(M, 1)``. An ``(M, 2)`` array is not accepted.
``f_vals`` then has to be all ``NaN``: a cached value comes without the noise
standard deviation that such an observation needs, and
:py:meth:`~pyvbmc.function_logger.FunctionLogger.add` takes the two
together.
