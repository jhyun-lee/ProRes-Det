"""Pipelines: offline needs only files, live needs a webcam and a projector."""

from .offline import FrameResult, process_sample, run_offline

__all__ = ["FrameResult", "process_sample", "run_offline", "run_live"]


def run_live(*args, **kwargs):
    """Lazy proxy: importing `live` pulls in ctypes/Win32 plumbing offline never needs."""
    from .live import run_live as _impl
    return _impl(*args, **kwargs)
