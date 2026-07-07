"""Logging configuration for the export CLI."""

import contextlib
import os
import sys

from loguru import logger


def _quiet_libraries(verbose: bool = False) -> None:
    """Silence noisy third-party output.

    PyAV surfaces libav/libx264 chatter (``non-existing PPS``, ``no frame!``,
    per-file libx264 encoder stats) on stderr. Those are expected during H.264
    priming and encoding and are not actionable, so we raise the threshold well
    above them unless the user asked for verbose output.
    """
    try:
        import av

        level = av.logging.WARNING if verbose else av.logging.FATAL
        av.logging.set_level(level)
        # LeRobot re-installs libav's default (direct-to-stderr) callback during
        # video encoding, which honors the C-level threshold rather than PyAV's
        # capture level — so set that too, or libx264 encoder stats leak through.
        if hasattr(av.logging, "set_libav_level"):
            av.logging.set_libav_level(level)
    except Exception:
        pass


# Clean, human-friendly console format for the CLI — no module:function:line noise.
_CLI_FORMAT = "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}"


def configure_cli_logging(verbose: bool = False) -> None:
    """Configure loguru + libraries for a tidy, customer-facing CLI.

    Default shows concise INFO progress; ``verbose`` restores DEBUG detail and
    library warnings for troubleshooting.

    Logs go to stdout so that :func:`suppress_native_stderr` can silence the
    C-level libav/libx264 chatter (which writes straight to fd 2) without
    swallowing our own output.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG" if verbose else "INFO",
        format=_CLI_FORMAT,
        colorize=True,
    )
    _quiet_libraries(verbose)


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    """Redirect the process's stderr file descriptor to /dev/null.

    LeRobot's video encoder swaps libav's log callback in a background thread, so
    the libx264 encoder-stats dump bypasses PyAV's log level and writes directly
    to fd 2. Silencing it reliably means redirecting the fd itself. Our own logs
    are on stdout (see :func:`configure_cli_logging`) and are unaffected. No-op
    when ``enabled`` is False (e.g. verbose mode), so nothing is hidden then.
    """
    if not enabled:
        yield
        return
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
