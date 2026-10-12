"""The default output of PyVBMC's loggers.

PyVBMC reports through named loggers (``"VBMC"``, ``"VBMC_init"``,
``"SVBMC"`` and a few others) and leaves the root logger alone, which belongs
to the application. Each of these loggers carries one shared handler that
writes the bare message to standard output while the application has
configured no logging, so that the iteration log shows by default. Once the
root logger has a handler, that handler writes nothing: the records reach
the application's handlers by propagation, formatted as it chose.
"""

import logging
import sys


class _StandardOutputHandler(logging.Handler):
    """Write each message to ``sys.stdout`` while the root logger has no
    handler.

    The stream is looked up at each record, so output follows a
    reassignment of ``sys.stdout``.
    """

    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        if logging.getLogger().handlers:
            return
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


_HANDLER = _StandardOutputHandler()


def get_logger(name):
    """Return the logger ``name``, carrying PyVBMC's default output handler.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logger : logging.Logger
        The logger, as ``logging.getLogger(name)`` returns it, with the
        handler added once.
    """
    logger = logging.getLogger(name)
    if _HANDLER not in logger.handlers:
        logger.addHandler(_HANDLER)
    return logger
