"""Running a check with no logging configured."""

import contextlib
import logging


@contextlib.contextmanager
def unconfigured_logging():
    """Run the block with no handler on the root logger.

    This is the state of a script or a notebook that configures no logging,
    which pytest's own handlers on the root logger hide from a test. On exit
    the root logger gets back the handlers it had and loses any that the
    block added.
    """
    root = logging.getLogger()
    saved = list(root.handlers)
    for handler in saved:
        root.removeHandler(handler)
    try:
        yield
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
