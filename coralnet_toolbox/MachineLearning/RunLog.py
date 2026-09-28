import re
import sys
import logging
import threading
from pathlib import Path
from contextlib import contextmanager


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Thread ident -> active _LogSinks (a nested capture adds another). Only threads with
# an active capture get their output copied.
_sinks = {}
_lock = threading.RLock()
_originals = None


class _ThreadRoutedStream:
    """
    Stands in for sys.stdout / sys.stderr while any capture is active. Every write still
    goes to the real stream, so the console is unchanged; writes made from a capturing
    thread are also copied to that thread's log file. Output from other threads (the GUI,
    other workers) never reaches the log.
    """
    def __init__(self, original):
        self._original = original

    def write(self, text):
        result = self._original.write(text)
        for sink in _sinks.get(threading.get_ident(), ()):
            sink.write(text)
        return result

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        # isatty, fileno, encoding, buffer, ... come from the real stream
        return getattr(self._original, name)


class _LogSink:
    """
    Writes captured text to a file the way a terminal would show it once finished:
    ANSI colour codes are dropped, and a carriage return lets the next text replace the
    current line, so a progress bar leaves only its final frame instead of every redraw.
    """
    def __init__(self, path):
        self._file = open(path, 'w', encoding='utf-8', errors='replace')
        self._line = ''
        self._carriage_return = False
        self._lock = threading.Lock()

    def write(self, text):
        text = _ANSI_ESCAPE.sub('', text)
        with self._lock:
            if self._file is None:
                return
            for token in re.split(r'(\r|\n)', text):
                if token == '\n':
                    self._file.write(self._line + '\n')
                    self._line = ''
                    self._carriage_return = False
                elif token == '\r':
                    self._carriage_return = True
                elif token:
                    if self._carriage_return:
                        self._line = token
                        self._carriage_return = False
                    else:
                        self._line += token
            self._file.flush()

    def close(self):
        with self._lock:
            if self._file is None:
                return
            if self._line:
                self._file.write(self._line + '\n')
            self._file.close()
            self._file = None


def _stream_handlers():
    """Every logging.StreamHandler currently attached to any logger."""
    loggers = [logging.getLogger()]
    loggers += [lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    for lg in loggers:
        for handler in list(lg.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                yield handler


def _set_stream(handler, stream):
    """
    Point a handler at stream. Some subclasses (e.g. logging's own last-resort
    handler) expose `stream` as a read-only property that looks up sys.stderr on
    every write; those already follow the proxy, so they are left alone.
    """
    try:
        handler.setStream(stream)
        return True
    except AttributeError:
        return False


def _install():
    """Route sys.stdout / sys.stderr (and loggers bound to them) through the proxy."""
    global _originals
    if _originals is not None:
        return
    stdout, stderr = sys.stdout, sys.stderr
    proxies = {id(stdout): _ThreadRoutedStream(stdout), id(stderr): _ThreadRoutedStream(stderr)}
    # Loggers such as Ultralytics' bind sys.stdout when the module is imported, so they
    # would bypass the proxy unless their handlers are pointed at it as well.
    retargeted = []
    for handler in _stream_handlers():
        stream = handler.stream
        proxy = proxies.get(id(stream))
        if proxy is not None and _set_stream(handler, proxy):
            retargeted.append((handler, stream))
    sys.stdout, sys.stderr = proxies[id(stdout)], proxies[id(stderr)]
    _originals = (stdout, stderr, retargeted)


def _uninstall():
    """Put the real streams back once no thread is capturing."""
    global _originals
    if _originals is None:
        return
    stdout, stderr, retargeted = _originals
    proxies = (sys.stdout, sys.stderr)
    for handler, stream in retargeted:
        if handler.stream in proxies:
            _set_stream(handler, stream)
    # Handlers created during the run (e.g. LightlyTrain's) captured the proxy itself
    for handler in _stream_handlers():
        if isinstance(handler.stream, _ThreadRoutedStream):
            _set_stream(handler, handler.stream._original)
    if isinstance(sys.stdout, _ThreadRoutedStream):
        sys.stdout = stdout
    if isinstance(sys.stderr, _ThreadRoutedStream):
        sys.stderr = stderr
    _originals = None


@contextmanager
def capture_run_log(log_path):
    """
    Save everything the calling thread prints (stdout, stderr and logging to either)
    to log_path, while still showing it in the console as normal.

    Only output from the calling thread is kept, so other work running at the same
    time does not end up in this log. Output from subprocesses (e.g. dataloader
    workers) is not captured.

    Args:
        log_path (str | Path): Text file to write; its folder is created if needed.
    """
    sink = None
    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sink = _LogSink(log_path)
    except Exception as e:
        print(f"Warning: Could not create log file {log_path}: {e}")

    if sink is None:
        yield None
        return

    ident = threading.get_ident()
    try:
        with _lock:
            _install()
            _sinks[ident] = _sinks.get(ident, ()) + (sink,)
    except Exception as e:
        # Logging is a convenience; never let it stop the run itself
        print(f"Warning: Could not capture console output to {log_path}: {e}")
        sink.close()
        yield None
        return

    try:
        yield log_path
    finally:
        with _lock:
            remaining = tuple(s for s in _sinks.get(ident, ()) if s is not sink)
            if remaining:
                _sinks[ident] = remaining
            else:
                _sinks.pop(ident, None)
            if not _sinks:
                _uninstall()
        sink.close()
