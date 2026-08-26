"""Smoke tests for coralnet_toolbox.

These exist because the previous check -- `import coralnet_toolbox` -- proved
almost nothing. `coralnet_toolbox/__init__.py` is six lines of metadata with no
imports, so that statement loads exactly one module and touches no dependency.
A build stayed green with PyQtAds, torch or rasterio completely broken; the
PyQtAds namespace difference between the PyPI and conda-forge builds passed CI
and only failed when the application was actually launched.

The tests below import the real GUI tree and build the main window, which is
what exercises those dependencies.
"""

import os
import unittest

# Must be set before PyQt5 initialises a platform plugin. CI runners have no
# display; the offscreen plugin needs none.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestPackageMetadata(unittest.TestCase):

    def test_version_is_exposed(self):
        import coralnet_toolbox
        self.assertRegex(coralnet_toolbox.__version__, r"^\d+\.\d+\.\d+")


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
                self.assertTrue(hasattr(ads, symbol), f"ads.{symbol} missing")


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
    """

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        from coralnet_toolbox.theme import apply_theme
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def _build_window(self):
        from coralnet_toolbox import __version__
        from coralnet_toolbox.QtMainWindow import MainWindow

        # MainWindow.__init__ ends with a "check for updates" call that GETs
        # pypi.org. A smoke test should not depend on the network, nor poll
        # PyPI on every push, so stub it for the duration of construction.
        original = MainWindow.open_check_for_updates_dialog
        MainWindow.open_check_for_updates_dialog = lambda self, *a, **k: None
        try:
            window = MainWindow(__version__)
        finally:
            MainWindow.open_check_for_updates_dialog = original

        self.addCleanup(window.close)
        return window

    def test_main_window_builds_with_docks(self):
        window = self._build_window()
        self.assertGreater(len(window.dock_manager.dockWidgetsMap()), 0)

    def test_layout_state_round_trips(self):
        dock_manager = self._build_window().dock_manager
        state = dock_manager.saveState()
        self.assertGreater(state.size(), 0)
        self.assertTrue(dock_manager.restoreState(state))


if __name__ == "__main__":
    unittest.main()
