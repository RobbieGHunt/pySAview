"""Responsive wrapping FlowLayout and compact control helpers for PyQt6 views."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLayout,
    QLayoutItem,
    QSizePolicy,
    QStyle,
    QWidget,
)


class FlowLayout(QLayout):
    """Left-to-right wrapping layout that flows items onto additional rows as width narrows.

    Key behaviours tailored for scientific control bars:
    - Hidden widgets are skipped cleanly (zero phantom gaps when optional controls toggle).
    - Items on each row are vertically centered against the tallest item in that row.
    - Items with an ``Expanding`` / ``MinimumExpanding`` horizontal size policy stretch to
      consume leftover horizontal space on their row (e.g. scrubbers and coordinate readouts).
    - Reports a minimal ``minimumSize()`` so parent splitters and top-level windows can
      compress freely both horizontally and vertically.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        margin: int = 0,
        h_spacing: int = -1,
        v_spacing: int = -1,
    ) -> None:
        super().__init__(parent)
        self._h_space = h_spacing
        self._v_space = v_spacing
        self._items: list[QLayoutItem] = []
        if margin >= 0:
            self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def horizontalSpacing(self) -> int:
        if self._h_space >= 0:
            return self._h_space
        return self._smart_spacing(QStyle.PixelMetric.PM_LayoutHorizontalSpacing)

    def verticalSpacing(self) -> int:
        if self._v_space >= 0:
            return self._v_space
        return self._smart_spacing(QStyle.PixelMetric.PM_LayoutVerticalSpacing)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, max(0, width), 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    @staticmethod
    def _item_effective_size(item: QLayoutItem) -> tuple[int, int]:
        sz = item.sizeHint()
        mn = item.minimumSize()
        mx = item.maximumSize()
        w = max(mn.width(), min(sz.width(), mx.width()))
        h = max(mn.height(), min(sz.height(), mx.height()))
        return max(0, w), max(0, h)

    def minimumSize(self) -> QSize:
        margins = self.contentsMargins()
        max_h = 0
        for item in self._visible_items():
            _, ih = self._item_effective_size(item)
            max_h = max(max_h, ih)
        return QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom() + max_h,
        )

    def _visible_items(self) -> list[QLayoutItem]:
        visible: list[QLayoutItem] = []
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            if item.isEmpty():
                continue
            visible.append(item)
        return visible

    def _smart_spacing(self, pm: QStyle.PixelMetric) -> int:
        parent = self.parent()
        if parent is None:
            return 6
        if isinstance(parent, QWidget):
            style = parent.style()
            if style is not None:
                val = style.pixelMetric(pm, None, parent)
                if val >= 0:
                    return val
            return 6
        if isinstance(parent, QLayout):
            return parent.spacing()
        return 6

    @staticmethod
    def _is_horizontally_ignored(item: QLayoutItem) -> bool:
        widget = item.widget()
        return widget is not None and widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored

    @classmethod
    def _is_horizontally_expanding(cls, item: QLayoutItem) -> bool:
        widget = item.widget()
        if widget is not None:
            policy = widget.sizePolicy().horizontalPolicy()
            return policy in (
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.MinimumExpanding,
                QSizePolicy.Policy.Ignored,
            )
        return bool(item.expandingDirections() & Qt.Orientation.Horizontal)

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        avail_w = max(0, effective.width())
        h_space = max(0, self.horizontalSpacing())
        v_space = max(0, self.verticalSpacing())

        visible = self._visible_items()
        if not visible:
            return margins.top() + margins.bottom()

        rows: list[list[tuple[QLayoutItem, int, int]]] = []
        cur_row: list[tuple[QLayoutItem, int, int]] = []
        row_x = 0

        for item in visible:
            eff_w, eff_h = self._item_effective_size(item)
            if self._is_horizontally_ignored(item):
                widget = item.widget()
                item_w = max(0, widget.minimumWidth()) if widget is not None else 0
            else:
                item_w = eff_w
            item_h = eff_h
            next_x = (row_x + h_space + item_w) if cur_row else item_w
            if cur_row and avail_w > 0 and next_x > avail_w:
                rows.append(cur_row)
                cur_row = [(item, item_w, item_h)]
                row_x = item_w
            else:
                cur_row.append((item, item_w, item_h))
                row_x = next_x
        if cur_row:
            rows.append(cur_row)

        y = effective.y()
        for row in rows:
            row_h = max((h for _, _, h in row), default=0)
            if not test_only:
                natural_w = sum(w for _, w, _ in row) + h_space * max(0, len(row) - 1)
                extra = avail_w - natural_w
                expanding_indices = [
                    idx for idx, (it, _, _) in enumerate(row) if self._is_horizontally_expanding(it)
                ]
                placed_widths = [w for _, w, _ in row]
                if extra > 0 and expanding_indices:
                    share, rem = divmod(extra, len(expanding_indices))
                    for k, idx in enumerate(expanding_indices):
                        placed_widths[idx] += share + (1 if k < rem else 0)
                elif extra < 0 and len(row) == 1 and avail_w > 0:
                    # Single oversized item on a narrow row: shrink into the available width
                    # (Qt still honors the widget's own minimumWidth if explicitly fixed).
                    placed_widths[0] = avail_w

                x = effective.x()
                for (item, _, item_h), placed_w in zip(row, placed_widths):
                    item_y = y + max(0, (row_h - item_h) // 2)
                    item.setGeometry(QRect(QPoint(x, item_y), QSize(placed_w, item_h)))
                    x += placed_w + h_space
            y += row_h + v_space

        total_h = (y - v_space - effective.y()) + margins.top() + margins.bottom()
        return max(margins.top() + margins.bottom(), total_h)


def make_flow_group(
    *widgets: QWidget,
    spacing: int = 6,
    expanding: bool = False,
) -> QWidget:
    """Group tightly coupled controls (e.g. a label and its combo) so they wrap as one unit."""
    container = QWidget()
    container.setObjectName("flowGroup")
    container.setStyleSheet("QWidget#flowGroup { background: transparent; }")
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        pol = widget.sizePolicy().horizontalPolicy()
        stretch = 1 if pol in (QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding) else 0
        layout.addWidget(widget, stretch)
    h_policy = QSizePolicy.Policy.Expanding if expanding else QSizePolicy.Policy.Preferred
    container.setSizePolicy(h_policy, QSizePolicy.Policy.Fixed)
    container.setMinimumSize(0, 0)
    return container


def configure_compact_combo(combo: QComboBox, min_chars: int = 10) -> QComboBox:
    """Keep a QComboBox compact horizontally without truncating its dropdown popup."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.setMinimumWidth(0)
    return combo


def fit_combo_popup(combo: QComboBox) -> None:
    """Ensure a compact QComboBox's popup view can display its longest item label cleanly."""
    if combo.count() <= 0:
        return
    fm = combo.fontMetrics()
    widest = max(fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count()))
    combo.view().setMinimumWidth(widest + 36)
