"""Checks on the logging objects a ``VBMC`` instance sets up.

The loggers are named after the package, not after the instance, so every
``VBMC`` in the process shares them with each other and with whatever the
user has configured on them.
"""

import logging
import os

import numpy as np
import pytest

from pyvbmc import VBMC

LOGGER_NAMES = ("VBMC", "VBMC_init", "VBMC.stream_only")


@pytest.fixture
def restored_loggers():
    """Leave the package's loggers with the handlers they came with."""
    saved = {
        name: list(logging.getLogger(name).handlers) for name in LOGGER_NAMES
    }
    yield
    for name, handlers in saved.items():
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)


def _vbmc(log_file, **options):
    D = 2
    return VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options={"log_file_name": str(log_file), "display": "off", **options},
        seed=1,
    )


def _file_handlers_for(logger, path):
    target = os.path.abspath(path)
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
        and handler.baseFilename == target
    ]


def test_logger_setup_leaves_no_stale_handler_for_the_log_file(
    tmp_path, restored_loggers
):
    """Setting up the logger leaves exactly one writer of the log file.

    The logger is shared across the process, so a second run writing to the
    same file must not find, or leave behind, handlers of an earlier one.
    """
    log_file = tmp_path / "run.log"
    vbmc = _vbmc(log_file)
    logger = logging.getLogger("VBMC")
    stale = [
        logging.FileHandler(filename=str(log_file), mode="a") for _ in range(2)
    ]
    for handler in stale:
        logger.addHandler(handler)

    vbmc._init_logger()

    try:
        assert not any(handler in logger.handlers for handler in stale)
        assert len(_file_handlers_for(logger, log_file)) == 1
    finally:
        for handler in stale:
            handler.close()


def test_logger_setup_keeps_handlers_of_other_kinds(
    tmp_path, restored_loggers
):
    """A user's own handler on the package logger is left alone."""
    log_file = tmp_path / "run.log"
    vbmc = _vbmc(log_file)
    logger = logging.getLogger("VBMC")
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    vbmc._init_logger()

    assert stream_handler in logger.handlers
    assert len(_file_handlers_for(logger, log_file)) == 1


def test_unknown_log_file_level_is_reported(tmp_path, restored_loggers):
    """An unusable file logging level is refused with a legible message."""
    log_file = tmp_path / "run.log"
    with pytest.raises(ValueError, match="log_file_level"):
        _vbmc(log_file, log_file_level="verbose")


@pytest.mark.parametrize(
    "level, expected",
    [
        ("off", logging.WARNING),
        ("iter", logging.INFO),
        ("full", logging.DEBUG),
        (logging.DEBUG, logging.DEBUG),
        (logging.DEBUG + 5, 15),
        (np.int64(25), 25),
        (logging.NOTSET, logging.NOTSET),
    ],
)
def test_log_file_is_written_at_any_logging_level(
    tmp_path, restored_loggers, level, expected
):
    """With ``log_file_name`` set the log file is written, at the level
    ``log_file_level`` gives: one of the three strings, or a level of the
    ``logging`` module, which takes any non-negative integer, custom levels
    and ``logging.NOTSET`` included."""
    log_file = tmp_path / "run.log"
    _vbmc(log_file, log_file_level=level)
    handlers = _file_handlers_for(logging.getLogger("VBMC_init"), log_file)
    assert len(handlers) == 1
    assert handlers[0].level == expected


@pytest.mark.parametrize(
    "level", [None, True, False, -1, 2.5, float(logging.INFO), "verbose"]
)
def test_log_file_level_that_is_no_level_is_refused(
    tmp_path, restored_loggers, level
):
    """Any other value of ``log_file_level`` is refused at construction,
    with a message that names the forms it takes."""
    log_file = tmp_path / "run.log"
    with pytest.raises(ValueError) as execinfo:
        _vbmc(log_file, log_file_level=level)
    message = execinfo.value.args[0]
    assert "log_file_level" in message
    assert '"iter"' in message and "non-negative integer" in message
