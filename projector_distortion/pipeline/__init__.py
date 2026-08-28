"""Pipelines: offline needs only files, live needs a webcam and a projector."""

from .offline import FrameResult, process_pair, process_sample, run_offline

__all__ = ["FrameResult", "process_pair", "process_sample", "run_offline",
           "run_live", "run_replay"]


def run_live(*args, **kwargs):
    """Lazy proxy: importing `live` pulls in ctypes/Win32 plumbing offline never needs."""
    from .live import run_live as _impl
    return _impl(*args, **kwargs)


def run_replay(*args, **kwargs):
    """Lazy proxy: `replay` reuses the live warp helpers, and so the same plumbing."""
    from .replay import run_replay as _impl
    return _impl(*args, **kwargs)
