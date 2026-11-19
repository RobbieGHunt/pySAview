"""Click-to-focus wheel guard so scroll wheels do not accidentally edit spinboxes/combos/sliders.

By default Qt gives spinboxes, comboboxes, and sliders ``WheelFocus`` so scrolling
a scrollable control panel hijacks the wheel and mutates numbers under the cursor.
This event filter enforces two rules application-wide:

1. Wheel events on ``QAbstractSpinBox``, ``QComboBox``, and ``QSlider`` are ignored
   (and bubble up to the enclosing ``QScrollArea``) unless that control currently
   has explicit keyboard focus from being clicked.
2. Clicking anywhere outside the currently focused spinbox/combobox/slider clears
   its focus immediately so subsequent scrolling scrolls the panel safely.
"""

from __future__ import annotations

import atexit
import weakref

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QSlider,
    QWidget,
)

_GUARDED_TYPES = (QAbstractSpinBox, QComboBox, QSlider)


def _enclosing_guarded(obj: QObject | None) -> QWidget | None:
    """Return the enclosing guarded control (or ``obj`` itself), or ``None``."""
    cur = obj
    while isinstance(cur, QWidget):
        if isinstance(cur, _GUARDED_TYPES):
            return cur
        cur = cur.parentWidget()
    return None


def _is_descendant_or_same(obj: QObject | None, ancestor: QWidget) -> bool:
    cur = obj
    while isinstance(cur, QObject):
        if cur is ancestor:
            return True
        cur = cur.parent()
    return False


class WheelFocusGuard(QObject):
    """Application- or widget-level filter enforcing click-to-scroll on numeric inputs."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        etype = event.type()

        if etype == QEvent.Type.Wheel:
            ctrl = _enclosing_guarded(obj)
            if ctrl is not None:
                focused = QApplication.focusWidget()
                if focused is None or not _is_descendant_or_same(focused, ctrl):
                    event.ignore()
                    return True
            return False

        if etype == QEvent.Type.MouseButtonPress:
            if not isinstance(obj, QWidget):
                return False
            focused = QApplication.focusWidget()
            if focused is not None:
                focused_ctrl = _enclosing_guarded(focused)
                if focused_ctrl is not None and not _is_descendant_or_same(obj, focused_ctrl):
                    focused.clearFocus()
                    focused_ctrl.clearFocus()
            return False

        return False


def enforce_strong_focus(root: QWidget) -> None:
    """Set ``StrongFocus`` on all guarded numeric/choice controls beneath ``root``."""
    for cls in _GUARDED_TYPES:
        for widget in root.findChildren(cls):
            if widget.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


def _detach_app_guard(app_ref: weakref.ref, guard_ref: weakref.ref) -> None:
    """Remove the global event filter before ``~QApplication()`` runs at interpreter exit."""
    try:
        app = app_ref()
        guard = guard_ref()
        if app is not None and guard is not None:
            app.removeEventFilter(guard)
            if getattr(app, "_pysaview_wheel_focus_guard", None) is guard:
                delattr(app, "_pysaview_wheel_focus_guard")
    except RuntimeError:
        pass


def install_wheel_focus_guard(root: QWidget) -> WheelFocusGuard:
    """Install strong-focus policies on ``root`` and ensure the global app guard is active."""
    enforce_strong_focus(root)
    app = QApplication.instance()
    if app is not None:
        existing = getattr(app, "_pysaview_wheel_focus_guard", None)
        if existing is None:
            existing = WheelFocusGuard(app)
            app.installEventFilter(existing)
            setattr(app, "_pysaview_wheel_focus_guard", existing)
            app_ref = weakref.ref(app)
            guard_ref = weakref.ref(existing)
            detach = lambda: _detach_app_guard(app_ref, guard_ref)
            app.aboutToQuit.connect(detach)
            atexit.register(detach)
        return existing
    guard = WheelFocusGuard(root)
    root.installEventFilter(guard)
    return guard
