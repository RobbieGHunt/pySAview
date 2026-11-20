# -*- mode: python ; coding: utf-8 -*-

import os
import sys

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
    # Matplotlib and its transitive stack (colormaps are pre-baked in resources/colormaps.npz)
    "matplotlib",
    "mpl_toolkits",
    "PIL",
    "fontTools",
    "kiwisolver",
    "contourpy",
    "cycler",
    "pyparsing",
    "dateutil",
    # Unused SciPy subpackages (exact intervals use only scipy.special)
    "scipy.cluster",
    "scipy.constants",
    "scipy.datasets",
    "scipy.differentiate",
    "scipy.fft",
    "scipy.fftpack",
    "scipy.integrate",
    "scipy.interpolate",
    "scipy.io",
    "scipy.misc",
    "scipy.ndimage",
    "scipy.odr",
    "scipy.optimize",
    "scipy.signal",
    "scipy.sparse",
    "scipy.spatial",
    "scipy.stats",
    # Unused Qt6 modules
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
entry_script = os.path.join(project_dir, 'pySAview', '__main__.py')
res_dir = os.path.join(project_dir, 'pySAview', 'resources')
example_dir = os.path.join(project_dir, 'pySAview', 'example_data')

datas = []
if os.path.exists(res_dir):
    datas.append((res_dir, 'pySAview/resources'))
if os.path.exists(example_dir):
    datas.append((example_dir, 'pySAview/example_data'))

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
        'scipy.special',
        'pySAview',
        'pySAview.core',
        'pySAview.core.reduction',
        'pySAview.ui',
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
    [],
    exclude_binaries=True,
    name='pySAview',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Keep False: GNU strip breaks auditwheel OpenBLAS alignment on Linux
    upx=False,
    console=False,  # Windowed GUI mode (no black console prompt)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='pySAview',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='pySAview.app',
        icon=None,
        bundle_identifier='fr.ill.superadam.pysaview',
        info_plist={
            'CFBundleDisplayName': 'pySAview',
            'CFBundleShortVersionString': '0.1.0',
            'NSHighResolutionCapable': True,
        },
    )
