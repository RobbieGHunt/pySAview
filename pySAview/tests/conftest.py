import pytest
from PyQt6.QtCore import QEvent, QSettings
from PyQt6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def isolate_preferences(tmp_path, monkeypatch):
    monkeypatch.setattr("pySAview.ui.main_window.QSettings",
                        lambda *a: QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat))


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    result = yield
    app = QApplication.instance()
    if app is not None:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
    return result
