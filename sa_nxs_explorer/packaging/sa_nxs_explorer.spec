# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os

block_cipher = None

excludes = [
    "tkinter",
    "pandas",
    "IPython",
    "jupyter",
    "notebook",
    "cv2",
    "torch",
    "pytest",
    "_pytest",
    "PyQt6.QtWebEngine",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtQml",
    "PyQt6.QtQuick",
    "PyQt6.QtQuickWidgets",
    "PyQt6.Qt3DCore",
    "PyQt6.Qt3DRender",
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtPdf",
    "PyQt6.QtPdfWidgets",
    "PyQt6.QtSql",
    "PyQt6.QtTest",
    "PyQt6.QtBluetooth",
    "PyQt6.QtNfc",
    "PyQt6.QtPositioning",
    "PyQt6.QtSensors",
    "PyQt6.QtSerialPort",
    "PyQt6.QtWebSockets",
]

project_dir = os.path.abspath(os.path.join(os.path.dirname(SPEC), '..', '..')) if 'SPEC' in globals() else os.path.abspath('.')
entry_script = os.path.join(project_dir, 'sa_nxs_explorer', '__main__.py')
res_dir = os.path.join(project_dir, 'sa_nxs_explorer', 'resources')

datas = []
if os.path.exists(res_dir):
    datas.append((res_dir, 'sa_nxs_explorer/resources'))

a = Analysis(
    [entry_script],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'h5py',
        'h5py.defs',
        'h5py.utils',
        'h5py._proxy',
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'sa_nxs_explorer',
        'sa_nxs_explorer.core',
        'sa_nxs_explorer.ui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SuperADAM_Explorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed GUI mode (no black console prompt)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
