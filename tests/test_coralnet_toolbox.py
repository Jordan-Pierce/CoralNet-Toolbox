"""Smoke tests for coralnet_toolbox.

These exist because the previous check -- `import coralnet_toolbox` -- proved
almost nothing. `coralnet_toolbox/__init__.py` is six lines of metadata with no
imports, so that statement loads exactly one module and touches no dependency.
A build stayed green with PyQtAds, torch or rasterio completely broken; the
PyQtAds namespace difference between the PyPI and conda-forge builds passed CI
and only failed when the application was actually launched.

The tests below import the real GUI tree and build the main window, which is
what exercises those dependencies.

Run these through `tests/run_tests.py`, not `unittest` directly -- see the note
in that file about Qt and interpreter shutdown.
"""

import os
import sys
import unittest

# Must be set before PyQt5 initialises a platform plugin. CI runners have no
# display; the offscreen plugin needs none.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Imported first on purpose, exactly as every real entry point does. The
# package pulls in torch before anything can load Qt, which on Windows is the
# difference between working and OSError [WinError 1114] on c10.dll. Importing
# PyQt5 here first would reintroduce the very bug the package guards against.
import coralnet_toolbox  # noqa: E402


class TestPackageMetadata(unittest.TestCase):

    def test_version_is_exposed(self):
        self.assertRegex(coralnet_toolbox.__version__, r"^\d+\.\d+\.\d+")

    def test_torch_is_loaded_by_the_package(self):
        """Guards the Windows load-order fix in coralnet_toolbox/__init__.py.

        If torch stops being imported there, Qt can win the race and c10.dll
        fails to initialise on Windows -- a failure that does not reproduce on
        Linux or macOS, so this assertion is the only thing that catches it.
        """
        self.assertIn("torch", sys.modules)


class TestDependencyImports(unittest.TestCase):
    """Third-party packages the toolbox cannot start without."""

    DEPENDENCIES = (
        "PyQt5.QtWidgets",
        "pyqtgraph",
        "numpy",
        "cv2",
        "rasterio",
        "geopandas",
        "torch",
        "ultralytics",
    )

    def test_dependencies_import(self):
        for name in self.DEPENDENCIES:
            with self.subTest(dependency=name):
                __import__(name)

    def test_ads_bindings_resolve(self):
        """The docking bindings are laid out differently per build.

        PyPI `PyQtAds` nests them in a submodule; the conda-forge `pyqtads`
        build exports the same names flat. QtAdsCompat hides that, and this
        asserts whichever build is installed actually yields the symbols.
        """
        from coralnet_toolbox.Layout.QtAdsCompat import ads
        for symbol in ("CDockManager", "CDockWidget", "CDockAreaWidget"):
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(ads, symbol), "ads." + symbol + " missing")


class TestImportSurface(unittest.TestCase):
    """The GUI import tree -- what a bare package import never reaches."""

    MODULES = (
        "coralnet_toolbox.main",
        "coralnet_toolbox.QtMainWindow",
        "coralnet_toolbox.Layout",
        "coralnet_toolbox.Layout.QtDockWrapper",
        "coralnet_toolbox.Layout.QtLayoutManager",
        "coralnet_toolbox.Rasters.VideoRaster",
        "coralnet_toolbox.MachineLearning.PreTrainModel.QtBase",
        "coralnet_toolbox.MachineLearning.TileDataset.QtBase",
    )

    def test_gui_modules_import(self):
        for name in self.MODULES:
            with self.subTest(module=name):
                __import__(name)


class TestMainWindow(unittest.TestCase):
    """Builds the real window offscreen.

    Catches breakage that survives a successful import: missing Qt platform
    plugins, docking symbols that exist but do not work, and layout
    serialization that no longer round-trips.

    One window is built for the whole class. Building several leaves multiple
    CDockManagers alive at once, which is not how the application runs and
    makes teardown considerably more fragile.
    """

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        from coralnet_toolbox.theme import apply_theme
        from coralnet_toolbox import __version__
        from coralnet_toolbox.QtMainWindow import MainWindow

        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

        # MainWindow.__init__ ends with a "check for updates" call that GETs
        # pypi.org. A smoke test should not depend on the network, nor poll
        # PyPI on every push, so stub it for the duration of construction.
        original = MainWindow.open_check_for_updates_dialog
        MainWindow.open_check_for_updates_dialog = lambda self, *a, **k: None
        try:
            cls.window = MainWindow(__version__)
        finally:
            MainWindow.open_check_for_updates_dialog = original

    @classmethod
    def tearDownClass(cls):
        # Destroy the window while the QApplication is still alive and the
        # interpreter is still healthy, rather than leaving it to shutdown.
        window, cls.window = cls.window, None
        window.close()
        window.deleteLater()
        cls.app.processEvents()

    def test_main_window_builds_with_docks(self):
        self.assertGreater(len(self.window.dock_manager.dockWidgetsMap()), 0)

    def test_layout_state_round_trips(self):
        dock_manager = self.window.dock_manager
        state = dock_manager.saveState()
        self.assertGreater(state.size(), 0)
        self.assertTrue(dock_manager.restoreState(state))
