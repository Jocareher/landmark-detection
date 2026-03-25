from __future__ import annotations

import io
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


class _TeeStream(io.TextIOBase):
    """Mirror writes to the original stream and a log file."""

    def __init__(self, primary_stream: TextIO, log_stream: TextIO) -> None:
        self.primary_stream = primary_stream
        self.log_stream = log_stream

    def write(self, text: str) -> int:
        self.primary_stream.write(text)
        self.log_stream.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary_stream.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        return self.primary_stream.isatty()


@contextmanager
def tee_terminal_output(log_path: str | Path) -> Iterator[Path]:
    """Duplicate terminal stdout and stderr into a log file."""
    resolved_log_path = Path(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        stdout_tee = _TeeStream(sys.stdout, log_stream)
        stderr_tee = _TeeStream(sys.stderr, log_stream)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield resolved_log_path
