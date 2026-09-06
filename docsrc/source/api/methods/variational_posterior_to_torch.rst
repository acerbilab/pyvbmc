=================================
``VariationalPosterior.to_torch``
=================================

.. automethod:: pyvbmc.VariationalPosterior.to_torch

The returned distribution is a copied snapshot with event shape ``(D,)``.
By default it represents the posterior in original coordinates using CPU
``torch.float64`` tensors. Explicit ``torch.float32`` and device exports are
also supported. The distribution's ``log_prob`` remains differentiable with
respect to its tensor input, but there is no gradient path back to the NumPy
posterior.

Conversion does not draw samples or alter the NumPy or torch random state.
Calling ``sample`` on the returned distribution uses torch's random generator.
For bounded original coordinates, density evaluation requires finite values
strictly inside the hard bounds; exact-bound and outside values fail support
validation.

See :ref:`Torch distribution` for a short example and installation details in
:ref:`optional integrations`.
