======================
``CalibrationProfile``
======================

An immutable record of the PDF and entropy chunk budgets used by a run.
Obtain a measured profile with :doc:`../functions/calibrate`, or construct
one explicitly to reproduce chosen settings. Profiles carry compact
provenance; detailed campaign measurements are kept separately in the cache.

.. code-block:: python

   from pyvbmc import CalibrationProfile, VariationalPosterior

   profile = CalibrationProfile(
       pdf_chunk_elements=65536,
       entropy_grad_chunk_elements=65536,
       entropy_value_chunk_elements=65536,
   )
   vp = VariationalPosterior(D=4, K=20, calibration=profile)

The budgets stay fixed for the lifetime of a resolved posterior. Copies,
saved posteriors and resumed VBMC runs preserve them. Machine compatibility
controls automatic cache reuse; it does not replace an explicitly supplied
or restored profile.

.. autoclass:: pyvbmc.CalibrationProfile
   :members:
