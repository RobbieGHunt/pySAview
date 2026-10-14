import h5py
import pytest
from sa_nxs_explorer.ui.main_window import MainWindow


def make_scans(tmp_path, count=12):
    paths = []
    for index in range(count):
        path = tmp_path / f'{index:06d}.nxs'
        with h5py.File(path, 'w') as file:
            entry = file.create_group('entry0')
            entry['run_number'] = index + 1
            entry['subtitle'] = 'LongUnbrokenSampleIdentifier' * 40
            variables = entry.create_group('data_scan/scanned_variables')
            variables['data'] = [[0, 1, 2], [10, 20, 30], [100, 200, 300]]
            names = variables.create_group('variables_names')
            names['name'] = [b'th', b'Detector1', b'Monitor1']
            names['unit'] = [b'deg', b'counts', b'counts']
        paths.append(path)
    return paths


def test_adding_scans_does_not_force_window_wider(qtbot, qapp, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qapp.processEvents()
    original_width = window.width()
    original_minimum = window.minimumSizeHint().width()
    for path in make_scans(tmp_path):
        assert window.load_file(path)
        qapp.processEvents()
        assert window.minimumSizeHint().width() <= original_minimum + 16
        assert window.width() <= original_width + 16
    assert '12 scans' in window.view_charts.lbl_title.text()
    assert len(window.view_charts.lbl_title.text()) < 160
    assert 'LongUnbrokenSampleIdentifier' in window.view_charts.lbl_title.toolTip()
    window.act_dark_mode.setChecked(True)
    qapp.processEvents()
    # Theme/font weight can alter the base chrome slightly, but scan content
    # must never turn a desktop window into a thousands-of-pixels-wide window.
    assert window.width() <= original_width + 200
    window.close()


def test_loading_before_first_show_keeps_toolbar_content_bounded(qtbot, qapp, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    paths = make_scans(tmp_path, 2)
    window.load_files(paths)
    window.show()
    qapp.processEvents()
    assert window.combo_active_file.minimumSizeHint().width() < 600
    assert len(window.view_charts.lbl_title.text()) < 160
    # Full descriptions survive in tooltips even though the displayed heading
    # and toolbar do not contribute their full text width to the layout.
    assert 'LongUnbrokenSampleIdentifier' in window.combo_active_file.itemData(0, 3)
    window.close()
