import pytest
from PyQt6.QtCore import QSettings


@pytest.fixture(autouse=True)
def isolate_preferences(tmp_path, monkeypatch):
    monkeypatch.setattr("pySAview.ui.main_window.QSettings",
                        lambda *a: QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat))
