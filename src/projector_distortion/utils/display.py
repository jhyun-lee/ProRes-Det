"""
Monitor enumeration and fullscreen window placement.

Split out of the live pipeline because none of it is about restoring or detecting
anything - it is the platform plumbing that gets one OpenCV window onto the
projector's display, and `collect.py` and `record.py` need it just as much.

On Windows, cv2.setWindowProperty(WND_PROP_FULLSCREEN) snaps the window back to the
primary display and silently undoes cv2.moveWindow(), which is why a naive --screen
has no effect. Geometry goes through the Win32 API instead.
"""

import ctypes
import sys
from collections import namedtuple
from typing import List, Optional

import cv2

Monitor = namedtuple("Monitor", "x y width height primary name")


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", _RECT), ("rcWork", _RECT),
                ("dwFlags", ctypes.c_ulong), ("szDevice", ctypes.c_wchar * 32)]


_MONITORINFOF_PRIMARY = 1


def _set_dpi_aware():
    """Report physical pixels; without this, display scaling corrupts coordinates."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _monitors_win32():
    user32 = ctypes.windll.user32
    proc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.POINTER(_RECT), ctypes.c_ssize_t)
    found = []

    def _cb(hmon, hdc, lprc, lparam):
        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(_MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            r = info.rcMonitor
            found.append(Monitor(r.left, r.top, r.right - r.left, r.bottom - r.top,
                                 bool(info.dwFlags & _MONITORINFOF_PRIMARY),
                                 info.szDevice))
        return 1

    user32.EnumDisplayMonitors(None, None, proc(_cb), 0)
    return found


def list_monitors() -> List[Monitor]:
    """All displays, primary first then left to right, so index 0 is always primary."""
    monitors = []
    if sys.platform == "win32":
        _set_dpi_aware()
        try:
            monitors = _monitors_win32()
        except Exception as e:
            print(f"warning: Win32 monitor enumeration failed ({e}); trying screeninfo.")
    if not monitors:
        try:
            from screeninfo import get_monitors
            monitors = [Monitor(m.x, m.y, m.width, m.height,
                                bool(getattr(m, "is_primary", False)), m.name or "")
                        for m in get_monitors()]
        except Exception as e:
            print(f"warning: screeninfo unavailable ({e}).")
    monitors.sort(key=lambda m: (not m.primary, m.x, m.y))
    return monitors


def _borderless_win32(title, mon):
    """Strip the frame and pin the window to `mon`. True when it took effect."""
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return False

    GWL_STYLE, GWL_EXSTYLE = -16, -20
    WS_POPUP, WS_VISIBLE = 0x80000000, 0x10000000
    FLAGS = 0x0020 | 0x0040 | 0x0010    # FRAMECHANGED | SHOWWINDOW | NOACTIVATE

    def signed(v):
        return v - (1 << 32) if v >= (1 << 31) else v

    user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_uint]
    user32.SetWindowLongW(hwnd, GWL_STYLE, signed(WS_POPUP | WS_VISIBLE))
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, 0)
    user32.SetWindowPos(hwnd, None, mon.x, mon.y, mon.width, mon.height, FLAGS)
    return True


def place_fullscreen(window, screen_index, image=None,
                     announce=True) -> Optional[Monitor]:
    """Put `window` fullscreen on `screen_index`. Returns that Monitor."""
    monitors = list_monitors()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    if not monitors:
        print("warning: no monitors detected; plain fullscreen on the primary display.")
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if image is not None:
            cv2.imshow(window, image)
            cv2.waitKey(1)
        return None

    if announce:
        print(f"{len(monitors)} monitor(s) detected:")
        for i, m in enumerate(monitors):
            star = " (primary)" if m.primary else ""
            # Just the index: demo.py --live takes it from rig.screen in live.yaml,
            # the data scripts from capture.screen / --screen.
            print(f"      screen {i} -> {m.width}x{m.height} at ({m.x},{m.y})"
                  f"{star}  {m.name}")

    if not 0 <= screen_index < len(monitors):
        print(f"warning: monitor {screen_index} unavailable; using 0 (primary).")
        screen_index = 0
    target = monitors[screen_index]

    cv2.setWindowProperty(window, cv2.WND_PROP_AUTOSIZE, 0)
    cv2.moveWindow(window, target.x, target.y)
    cv2.resizeWindow(window, target.width, target.height)
    if image is not None:
        cv2.imshow(window, image)     # the window must exist before Win32 can find it
        cv2.waitKey(1)

    if sys.platform == "win32":
        if not _borderless_win32(window, target):
            print("warning: could not restyle the window; using OpenCV fullscreen.")
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    return target
