"""Metadata and instrument motors dashboard for SuperADAM scans."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sa_nxs_explorer.core.models import MotorInfo, ScanData, ScanMetadata


class MetadataView(QWidget):
    """Scan experiment metadata and motor positioners inspection dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scan_data: ScanData | None = None
        self._all_motors: dict[str, MotorInfo] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ---------------- Top Cards: Experiment & Sample ----------------
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        # Card 1: Experiment Details
        grp_exp = QGroupBox("Experiment & Scan Details")
        exp_layout = QVBoxLayout(grp_exp)
        exp_layout.setSpacing(6)
        self.lbl_exp_details = QLabel("No file loaded.")
        self.lbl_exp_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_exp_details.setWordWrap(True)
        exp_layout.addWidget(self.lbl_exp_details)
        top_layout.addWidget(grp_exp, 1)

        # Card 2: Sample & Environment
        grp_sample = QGroupBox("Sample & Environment")
        sample_layout = QVBoxLayout(grp_sample)
        sample_layout.setSpacing(6)
        self.lbl_sample_details = QLabel("No sample metadata available.")
        self.lbl_sample_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_sample_details.setWordWrap(True)
        sample_layout.addWidget(self.lbl_sample_details)
        top_layout.addWidget(grp_sample, 1)

        splitter.addWidget(top_widget)

        # ---------------- Bottom: Motor Positions Table ----------------
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<b>SUPERADAM Motors & Positioners:</b>"))

        self.filter_motor = QLineEdit()
        self.filter_motor.setPlaceholderText("Filter motors by name...")
        self.filter_motor.textChanged.connect(self._filter_motors)
        header_row.addWidget(self.filter_motor)
        bottom_layout.addLayout(header_row)

        self.table_motors = QTableWidget(0, 5)
        self.table_motors.setHorizontalHeaderLabels([
            "Motor Name", "Current Value", "Target Value", "Offset Value", "Units"
        ])
        self.table_motors.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_motors.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_motors.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_motors.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_motors.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        bottom_layout.addWidget(self.table_motors, 1)

        splitter.addWidget(bottom_widget)
        splitter.setSizes([220, 500])

        main_layout.addWidget(splitter, 1)

    def set_data(self, scan_data: ScanData) -> None:
        """Assign scan data and render metadata cards and motor table."""
        self.scan_data = scan_data
        meta = scan_data.metadata

        # Format experiment card
        exp_lines = [
            f"<b>Run Number:</b> #{meta.run_number or 'N/A'}",
            f"<b>Title:</b> {meta.title or 'N/A'}",
            f"<b>Proposal:</b> {meta.proposal or 'N/A'}",
            f"<b>User:</b> {meta.user or 'N/A'} (Local Contact: {meta.local_contact or 'N/A'})",
            f"<b>Start Time:</b> {meta.start_time or 'N/A'}",
            f"<b>End Time:</b> {meta.end_time or 'N/A'}",
            f"<b>Duration:</b> {meta.duration:.1f} s" if meta.duration else "<b>Duration:</b> N/A",
            f"<b>Mode:</b> {meta.mode} (Preset: {meta.preset})",
            f"<b>Reactor Power:</b> {meta.reactor_power:.2f} kW" if meta.reactor_power else "<b>Reactor Power:</b> N/A",
        ]
        self.lbl_exp_details.setText("<br>".join(exp_lines))

        # Format sample & environment card
        sample_lines = [
            f"<b>Sample ID:</b> {meta.sample_id or 'N/A'}",
            f"<b>Formula:</b> {meta.chemical_formula or 'N/A'}",
            f"<b>Consistence:</b> {meta.sample_consistance or 'N/A'}",
            f"<b>Mass / Area:</b> {meta.sample_mass or 'N/A'} g / {meta.surface_area or 'N/A'} mm²",
        ]
        if meta.environment_variables:
            sample_lines.append("<b>Environment Sensors:</b>")
            for k, val in meta.environment_variables.items():
                sensor_name = k.split("/")[-1]
                sample_lines.append(f"  • {sensor_name}: {val:.4g}")
        self.lbl_sample_details.setText("<br>".join(sample_lines))

        # Motor table
        self._all_motors = scan_data.motors
        self._populate_motor_table(self._all_motors)

    def _populate_motor_table(self, motors: dict[str, MotorInfo]) -> None:
        self.table_motors.setRowCount(len(motors))
        for row, (name, m) in enumerate(sorted(motors.items())):
            self.table_motors.setItem(row, 0, QTableWidgetItem(name))

            val_str = f"{m.value:.5g}" if isinstance(m.value, (int, float)) else str(m.value or "--")
            tgt_str = f"{m.target_value:.5g}" if isinstance(m.target_value, (int, float)) else str(m.target_value or "--")
            off_str = f"{m.offset_value:.5g}" if isinstance(m.offset_value, (int, float)) else str(m.offset_value or "--")

            self.table_motors.setItem(row, 1, QTableWidgetItem(val_str))
            self.table_motors.setItem(row, 2, QTableWidgetItem(tgt_str))
            self.table_motors.setItem(row, 3, QTableWidgetItem(off_str))
            self.table_motors.setItem(row, 4, QTableWidgetItem(m.units or ""))

    def _filter_motors(self, text: str) -> None:
        query = text.strip().lower()
        if not query:
            for r in range(self.table_motors.rowCount()):
                self.table_motors.setRowHidden(r, False)
            return

        for r in range(self.table_motors.rowCount()):
            item = self.table_motors.item(r, 0)
            matches = query in item.text().lower() if item else False
            self.table_motors.setRowHidden(r, not matches)
