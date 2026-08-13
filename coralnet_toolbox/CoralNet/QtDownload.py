import os
import re
import sys
import time
import traceback
import ujson as json

from urllib.parse import urljoin

import concurrent
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox, QGroupBox,
                             QFormLayout, QApplication, QComboBox, QTextEdit,
                             QFileDialog, QSpinBox, QToolButton)

import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.IO.QtImportImages import SUPPORTED_IMAGE_EXTENSIONS

from coralnet_toolbox.Icons import get_icon

try:
    from urllib3.util.retry import Retry
except ImportError:  # urllib3 is a requests dependency, but never assume
    Retry = None


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# CoralNet's browse view paginates thumbnails 20 to a page.
BROWSE_PAGE_SIZE = 20

# Downloads are network-bound, not CPU-bound, so concurrency should track how many
# sockets the server tolerates — not how many cores the machine has. The previous
# os.cpu_count()-derived values both under-subscribed on typical hardware and
# evaluated to 0 (a hard ValueError from ThreadPoolExecutor) on 1-2 core machines.
DEFAULT_DOWNLOAD_WORKERS = 16
MAX_DOWNLOAD_WORKERS = 64

# (connect, read). Every request needs one: an un-timed-out request pins a worker
# thread forever if the server stops responding mid-transfer.
REQUEST_TIMEOUT = (10, 60)

# Per-chunk size when streaming an image to disk. 8 KiB meant ~1000 Python-level
# loop iterations for a single 8 MB image.
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# Cap on how many individual per-item failures are printed before the report
# switches to a single aggregate line. A source with 600 dead URLs should not
# bury the summary under 600 warnings.
MAX_REPORTED_FAILURES = 10


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def format_duration(seconds):
    """Render an elapsed time as a compact human-readable string."""
    if seconds is None:
        return "?"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_bytes(num_bytes):
    """Render a byte count using the largest unit that keeps it readable."""
    if not num_bytes:
        return "0 B"

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024


class ConsoleReporter:
    """
    Renders the download as a structured console report.

    Falls back to pure ASCII whenever the active stdout encoding cannot represent
    the box-drawing and status glyphs. Windows consoles frequently run a legacy
    code page, where printing them would raise UnicodeEncodeError and take down a
    download for a purely cosmetic reason.
    """

    WIDTH = 66
    LABEL_WIDTH = 15

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.fancy = self._supports_unicode(self.stream)

        if self.fancy:
            self.heavy_rule = "═"   # ═
            self.light_rule = "─"   # ─
            self.glyphs = {"ok": "✓", "fail": "✗",     # ✓ ✗
                           "skip": "·", "info": "→",   # · →
                           "warn": "!"}
            self.bullet = "·"
        else:
            self.heavy_rule = "="
            self.light_rule = "-"
            self.glyphs = {"ok": "OK", "fail": "XX",
                           "skip": "--", "info": "->",
                           "warn": "!!"}
            self.bullet = "-"

    @staticmethod
    def _supports_unicode(stream):
        """Return True when the stream's encoding can represent our glyph set."""
        encoding = getattr(stream, "encoding", None)
        if not encoding:
            return False
        try:
            "═─✓✗·→".encode(encoding)
            return True
        except (UnicodeEncodeError, LookupError):
            return False

    def _write(self, text=""):
        try:
            print(text, file=self.stream)
        except UnicodeEncodeError:
            # Belt and braces: never let formatting kill a download.
            print(text.encode("ascii", "replace").decode("ascii"), file=self.stream)

    def blank(self):
        self._write()

    def line(self, text):
        """Print a plain indented line."""
        self._write(f"  {text}")

    def rule(self, heavy=False):
        self._write((self.heavy_rule if heavy else self.light_rule) * self.WIDTH)

    def indented_rule(self):
        self._write("  " + self.light_rule * (self.WIDTH - 4))

    def header(self, title):
        """Print a heavy-ruled banner."""
        self.blank()
        self.rule(heavy=True)
        self._write(f"  {title}")
        self.rule(heavy=True)

    def step(self, label, status, text):
        """
        Print one aligned report row.

        Args:
            label (str): Left-hand phase name. Empty string continues the previous
                phase, so multi-stage work lines up under a single heading.
            status (str): Key into the glyph table ('ok', 'fail', 'skip', ...).
            text (str): Right-hand detail.
        """
        glyph = self.glyphs.get(status, self.glyphs["info"])
        self._write(f"  {label:<{self.LABEL_WIDTH}}{glyph:<3}{text}")

    def detail(self, text):
        """Print an unglyphed continuation line aligned under the detail column."""
        self._write(f"  {'':<{self.LABEL_WIDTH}}{'':<3}{text}")

    def join(self, *parts):
        """Join detail fragments with the separator bullet."""
        return f" {self.bullet} ".join(str(p) for p in parts if p)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


# TODO consider the use of prefix to filter the search space
class DownloadDialog(QDialog):
    """
    QDialog for downloading data from CoralNet sources.
    Provides UI for specifying source ID and download options.
    """

    def __init__(self, main_window):
        super(DownloadDialog, self).__init__(main_window)

        # Store reference to authentication dialog to get credentials
        self.main_window = main_window
        self.authentication_dialog = self.main_window.coralnet_authenticate_dialog

        # Initialize progress bar
        self.progress_bar = None

        # Initialize worker thread variables
        self.output_dir = None
        self.source_id = None
        self.source_dir = None
        self.download_options = None
        self.auth_token = None
        self.username = None
        self.password = None

        # Initialize driver
        self.driver = None
        self.headless = True
        self.logged_in = False

        # Shared authenticated HTTP session. Everything that does not strictly need
        # a real browser goes through this instead of Selenium, and every worker
        # reuses its connection pool rather than re-handshaking per request.
        self.session = None
        self.download_workers = DEFAULT_DOWNLOAD_WORKERS

        # Console report formatting + the page count stashed by whichever scan ran
        self.reporter = ConsoleReporter()
        self._last_page_count = None

        # Setup UI
        self.setWindowTitle("Download from CoralNet")
        self.resize(600, 400)  # Width, height reduced since we removed status section

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the source layout
        self.setup_source_layout()
        # Setup the options layout
        self.setup_options_layout()
        # Setup parameters layout
        self.setup_parameters_layout()
        # Setup buttons layout
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Download data from a CoralNet source. Specify the Source ID and select which items to download. To download data from multiple Sources, list them comma-separated in the Source ID field. The download will be saved to the specified Output Directory."
        )
        info_label.setWordWrap(True)
        info_label.setToolTip("Download images, annotations, metadata, and labelsets from CoralNet sources.\nMultiple sources can be specified (comma-separated).")
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_source_layout(self):
        """Setup the source ID input section."""
        source_group = QGroupBox("CoralNet Source")
        form_layout = QFormLayout()

        # Source ID input
        self.source_id_input = QLineEdit()
        self.source_id_input.setToolTip("CoralNet source ID(s) to download from.\nFor multiple sources: 1,2,3 (comma-separated, no spaces).")
        form_layout.addRow("Source ID:", self.source_id_input)

        # Output directory with browse button
        output_dir_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.select_output_dir)
        self.output_dir_input.setToolTip("Directory where downloaded data will be saved.\nSubdirectories will be created per source.")
        self.browse_button.setToolTip("Browse for an output directory.")
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(self.browse_button)
        form_layout.addRow("Output Directory:", output_dir_layout)

        # Set the form layout to the group box
        source_group.setLayout(form_layout)

        # Add the group box to the main layout
        self.layout.addWidget(source_group)

    def select_output_dir(self):
        """Open a directory selection dialog and update the output directory field."""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if directory:
            self.output_dir_input.setText(directory)

    def setup_options_layout(self):
        """Setup download options section."""
        options_group = QGroupBox("Download Options")
        form_layout = QFormLayout()

        # Dropdown for metadata
        self.metadata_dropdown = QComboBox()
        self.metadata_dropdown.addItems(["True", "False"])
        self.metadata_dropdown.setToolTip("Download source metadata (title, description, etc.).\nRequires less bandwidth than images.")
        form_layout.addRow("Download Metadata:", self.metadata_dropdown)

        # Dropdown for labelset
        self.labelset_dropdown = QComboBox()
        self.labelset_dropdown.addItems(["True", "False"])
        self.labelset_dropdown.setToolTip("Download labelset definitions and class mappings.\nNeeded to interpret annotation labels.")
        form_layout.addRow("Download Labelset:", self.labelset_dropdown)

        # Dropdown for annotations
        self.annotations_dropdown = QComboBox()
        self.annotations_dropdown.addItems(["True", "False"])
        self.annotations_dropdown.setToolTip("Download annotation data (labels, coordinates).\nRequired for training machine learning models.")
        form_layout.addRow("Download Annotations:", self.annotations_dropdown)

        # Dropdown for images
        self.images_dropdown = QComboBox()
        self.images_dropdown.addItems(["True", "False"])
        self.images_dropdown.setToolTip("Download actual image files from the source.\nLarge downloads can take significant time and disk space.")
        form_layout.addRow("Download Images:", self.images_dropdown)

        # Set the form layout to the group box
        options_group.setLayout(form_layout)

        # Add the group box to the main layout
        self.layout.addWidget(options_group)

    def setup_parameters_layout(self):
        """Setup the parameters section."""
        parameters_group = QGroupBox("Parameters")
        form_layout = QFormLayout()

        # Concurrent request count
        self.concurrency_input = QSpinBox()
        self.concurrency_input.setRange(1, MAX_DOWNLOAD_WORKERS)
        self.concurrency_input.setValue(DEFAULT_DOWNLOAD_WORKERS)
        self.concurrency_input.setToolTip(
            "Number of simultaneous HTTP requests used to scan pages and download images.\n"
            "Higher values finish faster; lower values are gentler on the server.\n"
            "The connection pool is sized to match this value."
        )
        form_layout.addRow("Concurrent Requests:", self.concurrency_input)

        # Image fetch rate input
        self.image_fetch_rate_input = QSpinBox()
        self.image_fetch_rate_input.setMinimum(3)
        self.image_fetch_rate_input.setValue(5)
        self.image_fetch_rate_input.setToolTip("Per-page delay for the fallback browser-driven page scan, in seconds.\nUnused when the faster HTTP scan succeeds.")
        form_layout.addRow("Image Fetch Rate (sec):", self.image_fetch_rate_input)

        # Image break time input
        self.fetch_break_time_input = QSpinBox()
        self.fetch_break_time_input.setMinimum(3)
        self.fetch_break_time_input.setValue(5)
        self.fetch_break_time_input.setToolTip("Recovery pause after a failed page fetch during the fallback browser scan, in seconds.\nUnused when the faster HTTP scan succeeds.")
        form_layout.addRow("Image Fetch Break Time (sec):", self.fetch_break_time_input)

        # Set the form layout to the group box
        parameters_group.setLayout(form_layout)

        # Add the group box to the main layout
        self.layout.addWidget(parameters_group)

    def setup_buttons_layout(self):
        """Setup the download and exit buttons"""
        button_layout = QHBoxLayout()

        # Add debug toggle button with bug icon
        self.debug_button = QToolButton()
        self.debug_button.setIcon(get_icon("www.svg"))
        self.debug_button.setToolTip("Toggle Headless Mode")
        self.debug_button.setCheckable(True)
        self.debug_button.setMaximumWidth(30)
        
        # 2. Define the handler
        def toggle_headless(checked):
            self.headless = not checked
        
        # 3. Connect the signal
        self.debug_button.toggled.connect(toggle_headless)
        
        # 4. Set the initial state (this will call toggle_headless once)
        self.debug_button.setChecked(not self.headless)

        button_layout.addWidget(self.debug_button)

        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self.start_download)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.reject)

        button_layout.addWidget(self.download_button)
        button_layout.addWidget(self.exit_button)

        # Add to main layout
        self.layout.addLayout(button_layout)

    # ------------------------------------------------------------------
    # HTTP session
    # ------------------------------------------------------------------

    def _build_session(self):
        """
        Create a logged-in requests.Session sized for the configured concurrency.

        One session (so one connection pool) is shared by every worker, which lets
        each request reuse a warm TCP+TLS connection instead of paying a fresh
        handshake. pool_maxsize must match the worker count: urllib3 defaults to 10
        and silently discards/reopens connections beyond that, which erases most of
        the benefit of raising the worker count.

        Returns:
            requests.Session: An authenticated session.
        """
        workers = max(1, int(getattr(self, 'download_workers', DEFAULT_DOWNLOAD_WORKERS)))
        session = requests.Session()

        retry = None
        if Retry is not None:
            retry_kwargs = dict(
                total=3,
                connect=3,
                read=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
            )
            try:
                # urllib3 >= 1.26
                retry = Retry(allowed_methods=frozenset(['GET', 'HEAD']), **retry_kwargs)
            except TypeError:
                # urllib3 < 1.26 spelled it method_whitelist
                retry = Retry(method_whitelist=frozenset(['GET', 'HEAD']), **retry_kwargs)

        adapter = HTTPAdapter(
            pool_connections=workers,
            pool_maxsize=workers,
            max_retries=retry if retry is not None else 3,
        )
        session.mount('https://', adapter)
        session.mount('http://', adapter)

        login_url = self.authentication_dialog.LOGIN_URL

        # Prime the session with the CSRF cookie, then post the login form.
        response = session.get(login_url, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")
        csrf_token = soup.find("input", attrs={"name": "csrfmiddlewaretoken"})

        if csrf_token is None:
            raise Exception("Could not find a CSRF token on the CoralNet login page")

        session.post(
            login_url,
            data={
                "username": self.username,
                "password": self.password,
                "csrfmiddlewaretoken": csrf_token["value"],
            },
            headers={"Referer": login_url},
            timeout=REQUEST_TIMEOUT,
        )

        return session

    def _get_session(self):
        """Return the shared authenticated session, creating it on first use."""
        if self.session is None:
            self.session = self._build_session()
        return self.session

    def _close_session(self):
        """Release the shared session and its pooled connections."""
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None

    @staticmethod
    def _wait_for_download(path, timeout=300, poll=0.5):
        """
        Block until a browser download has landed and stopped growing.

        Replaces a blind sleep: returns as soon as the file is actually complete
        (usually far sooner) and keeps waiting when the export is slow.

        Args:
            path (str): Expected final path of the downloaded file.
            timeout (int): Maximum seconds to wait.
            poll (float): Seconds between checks.

        Returns:
            bool: True if the file settled before the timeout.
        """
        partial = path + ".crdownload"
        deadline = time.time() + timeout

        while time.time() < deadline:
            QApplication.processEvents()

            if os.path.exists(path) and not os.path.exists(partial):
                size = os.path.getsize(path)
                if size > 0:
                    # Require the size to hold steady across one poll so we never
                    # hand back a file that is still being written.
                    time.sleep(poll)
                    if os.path.getsize(path) == size and not os.path.exists(partial):
                        return True
                    continue

            time.sleep(poll)

        return False

    def check_credentials(self):
        """Check if authentication credentials are available"""
        if not self.authentication_dialog or not self.authentication_dialog.is_authenticated():
            QMessageBox.warning(
                self,
                "Authentication Required",
                "Please authenticate with CoralNet first."
            )
            return False
        return True
                    
    def initialize_driver(self):
        """
        Initialize Chrome WebDriver with proper version handling and cross-platform support.
        Returns True if successful, False otherwise.
        """
        success = False

        # Initialize progress bar
        self.progress_bar.set_title("Initializing Chrome WebDriver")
        self.progress_bar.start_progress(100)

        try:
            # Create Chrome options with updated configurations
            options = webdriver.ChromeOptions()
            
            # Minimal logging but not complete silence for better diagnostics
            options.add_argument("--log-level=2")
            
            # Set download preferences
            prefs = {
                "download.default_directory": self.source_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,  # Keep security features enabled
                "profile.managed_default_content_settings.images": 2,
                "profile.managed_default_content_settings.stylesheet": 2,
                "profile.managed_default_content_settings.fonts": 2,
            }
            # Add preferences to options
            options.add_experimental_option("prefs", prefs)
            
            # Modern headless mode configuration
            if self.headless:
                # Use modern headless flag for Chrome v109+
                options.add_argument("--headless=new")
                # Prevent timeouts in headless mode
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                
            # Use Selenium 4.x Service approach for better driver management
            try:
                # Try Chrome for driver service with Selenium 4's improved manager
                from selenium.webdriver.chrome.service import Service as ChromeService
                
                # First try with the new Chrome Driver method (post Chrome v115)
                try:
                    from webdriver_manager.chrome import ChromeDriverManager
                    
                    # Try modern Chrome manager approach first (for Chrome v115+)
                    try:
                        from selenium.webdriver.chrome.service import Service as ChromeService
                        from webdriver_manager.core.os_manager import ChromeType
                        from webdriver_manager.chrome import ChromeDriverManager
                        
                        service = ChromeService(
                            ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
                        )
                        self.driver = webdriver.Chrome(service=service, options=options)
                        success = True
                        
                    # Fall back to traditional ChromeDriverManager for older versions
                    except (ImportError, Exception) as e:
                        self.progress_bar.update_progress_percentage(50)
                        self.progress_bar.setWindowTitle(f"Trying alternative driver setup: {e}")
                        service = ChromeService(ChromeDriverManager().install())
                        self.driver = webdriver.Chrome(service=service, options=options)
                        success = True
                        
                # Last resort: try finding local chromedriver in PATH
                except Exception as e:
                    self.progress_bar.update_progress_percentage(75)
                    self.progress_bar.setWindowTitle(f"Trying default driver: {e}")
                    self.driver = webdriver.Chrome(options=options)
                    success = True
                    
            # Handle older Selenium versions as fallback
            except ImportError:
                self.progress_bar.update_progress_percentage(85)
                self.progress_bar.setWindowTitle("Using legacy driver setup")
                # Fall back to the old-style initialization
                try:
                    # Cross-platform driver name handling
                    import platform
                    chrome_driver_name = "chromedriver.exe" if platform.system() == "Windows" else "chromedriver"
                    
                    # Try to find the driver in PATH first
                    from shutil import which
                    chrome_driver_path = which(chrome_driver_name)
                    
                    # If not found in PATH, use ChromeDriverManager
                    if not chrome_driver_path:
                        chrome_driver_path = ChromeDriverManager().install()
                    
                    self.driver = webdriver.Chrome(executable_path=chrome_driver_path, options=options)
                    success = True
                except Exception as local_e:
                    print(f"WARNING: Legacy driver setup failed: {str(local_e)}")
                    
        except Exception as e:
            error_message = f"ERROR: Could not initialize Chrome WebDriver: {str(e)}"
            print(error_message)
            traceback.print_exc()
            
        finally:
            self.progress_bar.finish_progress()
            
        if not success:
            print("\nTROUBLESHOOTING TIPS:")
            print("1. Ensure Google Chrome is installed and up-to-date")
            print("2. Check your internet connection (required for driver download)")
            print("3. Try running without headless mode for debugging")
            print("4. Check for corporate proxies or security software blocking WebDriver")
            
        return success

    def check_permissions(self):
        """
        Check the permissions of the current page.
        Returns the driver and status element if successful, raises exception otherwise.
        """
        status = None

        try:
            # Find the content container element
            path = "content-container"
            status = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, path)))

            # Check if status element has text
            if not status.text:
                raise Exception("Unable to access page information: element found but contains no text")

            # Check for specific error conditions
            if "Page could not be found" in status.text:
                raise Exception("Page could not be found: The requested source does not exist")
            elif "don't have permission" in status.text:
                raise Exception("Permission denied: You don't have permission to access this source")

        except Exception as e:
            # Propagate the exception with its original message
            raise Exception(f"Permission check failed: {str(e)}")

        return status

    def get_download_options(self):
        """Get the download options from the dropdowns"""
        options = {
            'images': self.images_dropdown.currentText() == "True",
            'labelset': self.labelset_dropdown.currentText() == "True",
            'annotations': self.annotations_dropdown.currentText() == "True",
            'metadata': self.metadata_dropdown.currentText() == "True"
        }
        return options

    def validate_inputs(self):
        """Validate the user inputs"""
        try:
            # Check if it's comma-separated list of source IDs
            [int(s.strip()) for s in self.source_id_input.text().strip().split(',')]
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Source IDs must be a numbers.")
            return False

        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Input Error", "Output directory is required.")
            return False

        options = self.get_download_options()
        if not any(options.values()):
            QMessageBox.warning(self, "Input Error", "Please select at least one download option.")
            return False

        self.image_fetch_rate = self.image_fetch_rate_input.value()
        self.fetch_break_time = self.fetch_break_time_input.value()
        self.download_workers = max(1, self.concurrency_input.value())

        return True

    def start_download(self):
        """Start the download process"""
        # Check if already authenticated
        if not self.check_credentials():
            return

        # Validate inputs
        if not self.validate_inputs():
            return

        # Get inputs
        source_ids = [int(s.strip()) for s in self.source_id_input.text().strip().split(',')]
        self.output_dir = os.path.normpath(self.output_dir_input.text().strip())
        self.download_options = self.get_download_options()

        # Get credentials from auth dialog
        self.auth_token = self.authentication_dialog.get_auth_token()
        self.username = self.authentication_dialog.username_input.text()
        self.password = self.authentication_dialog.password_input.text()

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.progress_bar = ProgressBar(self, "CoralNet Download")
        self.progress_bar.show()

        # Track the source directories that were successfully downloaded to, for optional import
        self.downloaded_source_dirs = []

        download_succeeded = False

        run_started = time.perf_counter()
        total_images = 0
        completed_sources = 0

        try:
            for index, source_id in enumerate(source_ids, start=1):
                self.progress_bar.set_title(f"Downloading Data from Source {source_id}")
                self.source_id = source_id

                position = f"  {self.reporter.bullet}  source {index} of {len(source_ids)}" \
                    if len(source_ids) > 1 else ""
                self.reporter.header(f"CoralNet Source {source_id}{position}")

                # Start the download process for this source ID
                total_images += self.download()
                completed_sources += 1
                self.downloaded_source_dirs.append(self.source_dir)

            download_succeeded = True

        except Exception as e:
            self.reporter.blank()
            self.reporter.step("Aborted", "fail", f"{e}")
            QMessageBox.critical(self, "Download Error", f"{str(e)}")

        finally:
            self._report_run_summary(completed_sources, len(source_ids),
                                     total_images, time.perf_counter() - run_started)

            # Make cursor not busy
            QApplication.restoreOverrideCursor()

            if self.progress_bar:
                self.progress_bar.finish_progress()
                self.progress_bar.close()
                self.progress_bar = None

            if self.driver:
                self.driver.quit()
                self.driver = None

            self._close_session()
            self.logged_in = False

        # Show the completion dialog (and optionally import) only after the
        # download progress bar has been fully closed, to avoid overlapping dialogs
        if download_succeeded:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Download Complete")
            msg_box.setText("Download completed successfully.")
            msg_box.setIcon(QMessageBox.Information)
            ok_button = msg_box.addButton(QMessageBox.Ok)
            import_button = msg_box.addButton("Import Data", QMessageBox.ActionRole)
            msg_box.setDefaultButton(ok_button)
            msg_box.exec_()

            if msg_box.clickedButton() == import_button:
                # Close the download dialog itself before importing, so it's not
                # left hanging around behind the import progress bars
                self.accept()
                self.import_downloaded_data()

    def _report_run_summary(self, completed, requested, total_images, elapsed):
        """Print the closing banner covering every source in this run."""
        if requested == 0:
            return

        if completed == requested:
            title = "Download complete"
        elif completed == 0:
            title = "Download failed"
        else:
            title = f"Download finished with errors ({completed}/{requested} sources)"

        parts = [title]
        if requested > 1:
            parts.append(f"{completed} sources")
        parts.append(f"{total_images} images")
        parts.append(format_duration(elapsed))

        self.reporter.blank()
        self.reporter.rule(heavy=True)
        self.reporter.line(self.reporter.join(*parts))
        self.reporter.rule(heavy=True)
        self.reporter.blank()

    def download(self):
        """
        Run the download process for the current source.

        Returns:
            int: Number of images newly downloaded, for the run summary.
        """
        # Create source directory (normalized path needed for Selenium)
        self.source_dir = os.path.normpath(os.path.join(os.path.abspath(self.output_dir),
                                                        str(self.source_id)))
        os.makedirs(self.source_dir, exist_ok=True)

        started = time.perf_counter()

        # Initialize the driver
        if not self.driver:
            if not self.initialize_driver():
                raise Exception("Failed to find a supported browser (see console log)")

        # Login to CoralNet
        if not self.logged_in:
            if not self.login():
                raise Exception("Failed to login to CoralNet (see console log)")

        # Check permissions
        if not self.check_permissions():
            raise Exception("Failed to permissions check (see console log)")

        # Download metadata if selected
        if self.download_options.get('metadata', False):
            self.download_metadata()
        else:
            self.reporter.step("Metadata", "skip", "not selected")

        # Download labelset if selected
        if self.download_options.get('labelset', False):
            self.download_labelset()
        else:
            self.reporter.step("Labelset", "skip", "not selected")

        # Download annotations if selected
        if self.download_options.get('annotations', False):
            self.download_annotations()
        else:
            self.reporter.step("Annotations", "skip", "not selected")

        downloaded_count = 0

        # Download images if selected
        if self.download_options.get('images', False):
            scan_started = time.perf_counter()
            images, success = self.get_images()
            scan_elapsed = time.perf_counter() - scan_started

            if not success:
                raise Exception("Failed while scanning for images (see console log)")

            pages = self._last_page_count
            page_text = f"{pages} pages" if pages else "1 page"
            self.reporter.step("Images", "ok", self.reporter.join(
                f"{len(images)} found across {page_text}", format_duration(scan_elapsed)))

            if len(images):
                # Get image URLs for each of the images
                url_started = time.perf_counter()
                images['Image URL'] = self.get_image_urls(images['Image Page'].tolist())
                url_elapsed = time.perf_counter() - url_started

                resolved = int(images['Image URL'].notna().sum())
                self.reporter.step("", "ok" if resolved == len(images) else "warn",
                                   self.reporter.join(
                                       f"{resolved}/{len(images)} URLs resolved",
                                       format_duration(url_elapsed)))

                # Download images
                dl_started = time.perf_counter()
                stats = self.download_images(images)
                dl_elapsed = time.perf_counter() - dl_started

                downloaded_count = stats['downloaded']

                parts = [f"{stats['downloaded']}/{len(images)} downloaded",
                         format_duration(dl_elapsed)]
                if stats['bytes']:
                    parts.append(format_bytes(stats['bytes']))
                if dl_elapsed > 0 and stats['downloaded']:
                    parts.append(f"{stats['downloaded'] / dl_elapsed:.1f} img/s")

                status = "ok" if stats['failed'] == 0 else "warn"
                self.reporter.step("", status, self.reporter.join(*parts))

                if stats['skipped']:
                    self.reporter.detail(f"{stats['skipped']} already present, skipped")
                if stats['failed']:
                    self.reporter.detail(f"{stats['failed']} failed")
        else:
            self.reporter.step("Images", "skip", "not selected")

        self.reporter.indented_rule()
        self.reporter.step("Done", "info", self.reporter.join(
            format_duration(time.perf_counter() - started), self.source_dir))

        return downloaded_count

    def login(self):
        """
        Log in to CoralNet using Selenium.
        """
        # Create a variable for success
        success = False

        # Get auth info from the authentication dialog
        username = self.authentication_dialog.username_input.text()
        password = self.authentication_dialog.password_input.text()

        # Add credentials to driver capabilities for later use
        self.driver.capabilities['credentials'] = {
            'username': username,
            'password': password
        }

        # Initialize progress bar
        self.progress_bar.set_title("Logging into CoralNet")
        self.progress_bar.start_progress(100)

        try:
            # Navigate to the page to login
            self.driver.get(self.authentication_dialog.CORALNET_URL + "/accounts/login/")

            # Find the username button
            path = "id_username"
            username_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, path)))

            # Find the password button
            path = "id_password"
            password_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, path)))

            # Find the login button. Waiting on clickability (rather than mere
            # presence followed by a blind 3-second sleep) returns as soon as the
            # form is actually usable.
            path = "//input[@type='submit'][@value='Sign in']"
            login_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, path)))

            # Enter the username and password
            username_input.send_keys(self.driver.capabilities['credentials']['username'])
            password_input.send_keys(self.driver.capabilities['credentials']['password'])

            # Click the login button
            login_button.click()

            # Confirm login was successful; after 10 seconds, throw an error.
            path = "//button[text()='Sign out']"

            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, path)))

            # Login was successful
            success = True
            self.logged_in = True

        except Exception as e:
            self.reporter.step("Login", "fail", f"could not sign in as {username}: {e}")

        finally:
            self.progress_bar.finish_progress()

        return success

    def download_metadata(self):
        """
        Given a source ID, download the labelset.
        """
        success = False

        # To hold the metadata
        meta = []

        # Initialize progress bar
        self.progress_bar.set_title("Downloading Metadata")
        self.progress_bar.start_progress(100)

        try:
            # Go to the meta page
            self.driver.get(self.authentication_dialog.CORALNET_URL + f"/source/{self.source_id}/")

            # Convert the page to soup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            if soup is None:
                raise Exception("Unable to parse the page source")

            script = None

            # Of the scripts, find the one containing model metadata
            for script in soup.find_all("script"):
                if "Classifier overview" in script.text:
                    script = script.text
                    break

            if not script:
                self.reporter.step("Metadata", "skip", "no classifier history")
                success = True  # Nothing to download, exit early

            else:
                # Parse the data when represented as a string, convert to dict
                start_index = script.find("let classifierPlotData = ") + len("let classifierPlotData = ")
                end_index = script.find("];", start_index) + 1  # Adding 1 to include the closing bracket

                # Extract the substring containing the data
                classifier_plot_data_str = script[start_index:end_index]

                # Convert single quotes to double quotes for JSON compatibility
                classifier_plot_data_str = classifier_plot_data_str.replace("'", '"')

                # Parse the string into a Python list of dictionaries
                data = json.loads(classifier_plot_data_str)

                # Loop through and collect meta from each model instance, store
                for idx, point in enumerate(data):
                    classifier_nbr = point["x"]
                    score = point["y"]
                    nimages = point["nimages"]
                    traintime = point["traintime"]
                    date = point["date"]
                    src_id = point["pk"]

                    meta.append([classifier_nbr,
                                score,
                                nimages,
                                traintime,
                                date,
                                src_id])

                # Convert list to dataframe
                meta = pd.DataFrame(meta, columns=['Classifier nbr',
                                                   'Accuracy',
                                                   'Trained on',
                                                   'Date',
                                                   'Traintime',
                                                   'Global id'])

                # Save to disk
                meta_path = os.path.join(self.source_dir, "metadata.csv")
                meta.to_csv(meta_path)

                # Check that it was saved
                if os.path.exists(meta_path):
                    self.reporter.step("Metadata", "ok", self.reporter.join(
                        "saved", f"{len(meta)} classifiers",
                        format_bytes(os.path.getsize(meta_path))))
                    success = True
                else:
                    raise Exception("Metadata could not be saved")

        except Exception as e:
            self.reporter.step("Metadata", "fail", f"{e}")

        finally:
            self.progress_bar.finish_progress()

        return success

    def download_labelset(self):
        """
        Given a source ID, download the labelset.
        """
        success = False

        # To hold the labelset
        labelset = None

        # Initialize progress bar
        self.progress_bar.set_title("Downloading Labelset")
        self.progress_bar.start_progress(100)

        try:
            # Go to the images page
            self.driver.get(self.authentication_dialog.CORALNET_URL + f"/source/{self.source_id}/labelset/")

            # Get the page source HTML
            html_content = self.driver.page_source
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')
            # Find the table with id 'label-table'
            table = soup.find('table', {'id': 'label-table'})
            
            if table is None:
                raise Exception("Unable to find the label table in the page source")

            if not table.find_all('tr'):
                self.reporter.step("Labelset", "skip", "no labels defined")
                success = True  # Nothing to download, exit early

            else:
                # Initialize lists to store data
                label_ids = []
                names = []
                short_codes = []

                # Loop through each row in the table
                for idx, row in enumerate(table.find_all('tr')):
                    
                    # Skip the header row
                    if not row.find('th'):
                        # Extract label ID from href attribute of the anchor tag
                        label_id = row.find('a')['href'].split('/')[-2]
                        label_ids.append(label_id)
                        # Extract Name from the anchor tag
                        name = row.find('a').text.strip()
                        names.append(name)
                        # Extract Short Code from the second td tag
                        short_code = row.find_all('td')[1].text.strip()
                        short_codes.append(short_code)

                # Create a pandas DataFrame
                labelset = pd.DataFrame({
                    'Label ID': label_ids,
                    'Name': names,
                    'Short Code': short_codes
                })

                # Save the labelset as a CSV file
                labelset_path = os.path.join(self.source_dir, "labelset.csv")
                labelset.to_csv(labelset_path)

                # Check that it was saved
                if os.path.exists(labelset_path):
                    self.reporter.step("Labelset", "ok", self.reporter.join(
                        "saved", f"{len(labelset)} labels"))
                    success = True
                else:
                    raise Exception("Labelset could not be saved")

        except Exception as e:
            self.reporter.step("Labelset", "fail", f"{e}")

        finally:
            self.progress_bar.finish_progress()

        return success

    def download_annotations(self):
        """
        This function downloads the annotations from a CoralNet source.
        """
        success = False

        # Initialize progress bar
        self.progress_bar.set_title("Downloading Annotations")
        self.progress_bar.start_progress(100)

        try:
            # Navigate to the source browse images page
            self.driver.get(self.authentication_dialog.CORALNET_URL + f"/source/{self.source_id}/browse/images/")

            # Find and interact with the export dropdown
            browse_action_dropdown = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "browse_action"))
            )

            # Select the "Export Annotations, CSV" option from the dropdown
            select = Select(browse_action_dropdown)
            select.select_by_value("export_annotations")

            # Select "All images" from the dropdown
            image_select_dropdown = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "image_select_type"))
            )
            select = Select(image_select_dropdown)
            select.select_by_value("all")

            # Select "Both" for the label format
            both_option = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='label_format'][value='both']"))
            )
            both_option.click()

            # Select all optional columns
            optional_columns = self.driver.find_elements(By.CSS_SELECTOR, "input[name='optional_columns']")
            for checkbox in optional_columns:
                # Current criteria for finding the right checkboxes
                if checkbox.accessible_name and checkbox.aria_role != 'none':
                    checkbox.click()

            # Wait for the options to be selected
            time.sleep(1)

            # Find and click the Go button
            go_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//form[@id='export-annotations-prep-form']//button"))
            )
            go_button.click()

            # Bound the server-side prep wait instead of spinning forever.
            prep_deadline = time.time() + 600
            while "Working" in go_button.accessible_name and time.time() < prep_deadline:
                QApplication.processEvents()
                time.sleep(1)

            # Wait for the file to actually land rather than sleeping a fixed 10s
            annotations_path = os.path.join(self.source_dir, "annotations.csv")

            if self._wait_for_download(annotations_path):
                self.reporter.step("Annotations", "ok", self.reporter.join(
                    "saved", format_bytes(os.path.getsize(annotations_path))))
                success = True
            else:
                raise Exception("export did not complete in time")

        except Exception as e:
            self.reporter.step("Annotations", "fail", f"{e}")

        finally:
            self.progress_bar.finish_progress()

        return success

    @property
    def browse_url(self):
        """URL of the current source's browse-images view."""
        return self.authentication_dialog.CORALNET_URL + f"/source/{self.source_id}/browse/images/"

    @staticmethod
    def _parse_browse_page(html, base_url):
        """
        Extract thumbnail names and absolute image-page URLs from one browse page.

        Pairs each name with its URL inside the same .thumb_wrapper rather than
        zipping two independent element lists, so a wrapper missing either half
        cannot shift every subsequent name onto the wrong URL.

        Args:
            html (str): Raw page HTML.
            base_url (str): URL the page was fetched from, for resolving relative hrefs.

        Returns:
            tuple: (names, page_urls, next_page_href or None)
        """
        soup = BeautifulSoup(html, 'html.parser')

        names = []
        page_urls = []

        for wrapper in soup.select('.thumb_wrapper'):
            anchor = wrapper.find('a')
            image = wrapper.find('img')

            if anchor is None or image is None:
                continue

            href = anchor.get('href')
            if not href:
                continue

            # Selenium's get_attribute('href') returned absolute URLs; BeautifulSoup
            # hands back the raw attribute, so resolve it here instead.
            page_urls.append(urljoin(base_url, href))
            names.append(image.get('alt') or '')

        next_anchor = soup.find('a', attrs={'title': 'Next page'})
        next_href = None
        if next_anchor is not None and next_anchor.get('href'):
            next_href = urljoin(base_url, next_anchor['href'])

        return names, page_urls, next_href

    @staticmethod
    def _parse_total_images(html):
        """Read the total image count from the browse page's summary line."""
        soup = BeautifulSoup(html, 'html.parser')
        line = soup.select_one('div.line')

        if line is None:
            return None

        numbers = re.findall(r'\d[\d,]*', line.get_text())
        if not numbers:
            return None

        return int(numbers[-1].replace(',', ''))

    def _scan_images_via_requests(self):
        """
        Fetch every browse page concurrently over plain HTTP.

        The browser-driven scan visits pages strictly serially and sleeps
        image_fetch_rate seconds on each one, so a 1000-image source spent minutes
        idle before a single byte of imagery moved. These pages are static HTML
        behind the same session cookie, so they can be fetched in parallel.

        Deliberately conservative: any page that fails to fetch, or any hint that
        the pagination scheme is not what we inferred, aborts the whole fast path
        and defers to Selenium. A partially-scanned source would silently download
        a truncated image set, which is far worse than being slow.

        Returns:
            tuple or None: (names, page_urls), or None to fall back to Selenium.
        """
        session = self._get_session()
        browse_url = self.browse_url

        response = session.get(browse_url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return None

        first_names, first_urls, next_href = self._parse_browse_page(response.text, browse_url)
        if not first_urls:
            return None

        total_images = self._parse_total_images(response.text)
        if total_images is None:
            return None

        # Ceiling division; the old `// 20 + 1` requested a spurious empty page
        # whenever the count was an exact multiple of the page size.
        total_pages = max(1, -(-total_images // BROWSE_PAGE_SIZE))
        self._last_page_count = total_pages

        if total_pages == 1:
            return first_names, first_urls

        # Learn the pagination parameter from the real "Next page" link rather than
        # hard-coding one. If the link is not shaped the way we expect, bail out.
        page_param = None
        if next_href:
            match = re.search(r'[?&](\w+)=2(?:&|$)', next_href)
            if match:
                page_param = match.group(1)

        if page_param is None:
            return None

        def fetch_page(page_number):
            try:
                page_response = session.get(
                    browse_url,
                    params={page_param: page_number},
                    timeout=REQUEST_TIMEOUT,
                )
                if page_response.status_code != 200:
                    return page_number, [], [], False
                names, urls, _ = self._parse_browse_page(page_response.text, browse_url)
                return page_number, names, urls, True
            except Exception as e:
                self.reporter.detail(f"page {page_number} failed: {e}")
                return page_number, [], [], False

        pages = {1: (first_names, first_urls)}
        workers = min(self.download_workers, total_pages - 1)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(fetch_page, p) for p in range(2, total_pages + 1)]

            completed = 0
            last_percent = -1

            for future in concurrent.futures.as_completed(futures):
                page_number, names, urls, ok = future.result()

                if not ok:
                    self.reporter.detail("page scan incomplete, retrying via browser")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return None

                pages[page_number] = (names, urls)

                completed += 1
                percent = int(completed / len(futures) * 100)
                if percent != last_percent:
                    self.progress_bar.update_progress_percentage(percent)
                    last_percent = percent

        # Sanity check the inferred pagination: if page 2 came back identical to
        # page 1, the parameter was ignored and every page is really page 1.
        if pages.get(2, ([], []))[1][:1] == first_urls[:1]:
            self.reporter.detail("pagination not recognised, retrying via browser")
            return None

        names = []
        page_urls = []
        for page_number in sorted(pages):
            page_names, urls = pages[page_number]
            names.extend(page_names)
            page_urls.extend(urls)

        return names, page_urls

    def _scan_images_via_selenium(self):
        """
        Walk the browse pages one at a time in the real browser.

        Fallback for when the HTTP scan cannot run. Still paginates serially, but
        waits on the thumbnails actually appearing instead of sleeping blindly.

        Returns:
            tuple: (names, page_urls)
        """
        self.driver.get(self.browse_url)

        try:
            page_element = self.driver.find_element(By.CSS_SELECTOR, 'div.line')
            total_images = int(re.findall(r'\d[\d,]*', page_element.text)[-1].replace(',', ''))
            total_pages = max(1, -(-total_images // BROWSE_PAGE_SIZE))
        except Exception:
            raise Exception("Could not determine total amount of images; please report this issue")

        self._last_page_count = total_pages

        image_page_urls = []
        image_names = []

        current_page = 1
        has_next_page = True

        while has_next_page and current_page <= total_pages:
            try:
                # Wait for this page's thumbnails rather than sleeping a fixed
                # image_fetch_rate seconds regardless of how fast the page loaded.
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.thumb_wrapper a'))
                )
                url_elements = self.driver.find_elements(By.CSS_SELECTOR, '.thumb_wrapper a')
                name_elements = self.driver.find_elements(By.CSS_SELECTOR, '.thumb_wrapper img')
            except Exception:
                self.reporter.detail(f"page slow to load, pausing {self.fetch_break_time}s")
                time.sleep(self.fetch_break_time)
                continue

            first_thumb = url_elements[0] if url_elements else None

            # Iterate over the image elements
            for url_element, name_element in zip(url_elements, name_elements):
                # Extract the href attribute (URL)
                image_page_urls.append(url_element.get_attribute('href'))
                # Extract the title attribute (image name)
                image_names.append(name_element.get_attribute('alt'))

            try:
                next_button = self.driver.find_element(By.CSS_SELECTOR, 'a[title="Next page"]')
            except Exception:
                break

            if not (next_button.is_displayed() and next_button.is_enabled()):
                break

            next_button.click()
            current_page += 1

            # Confirm the click actually navigated before scraping again, so we
            # cannot scrape the same page twice.
            if first_thumb is not None:
                try:
                    WebDriverWait(self.driver, 30).until(EC.staleness_of(first_thumb))
                except Exception:
                    self.reporter.detail("page did not advance, stopping scan")
                    break

            self.progress_bar.update_progress_percentage(
                min(100, int(current_page / total_pages * 100))
            )

        return image_names, image_page_urls

    def get_images(self):
        """
        Given a source ID, retrieve the image names and page URLs.
        Returns a DataFrame containing image names and their page URLs.
        """
        # Initialize result variables
        images = []
        success = False

        # Initialize progress bar
        self.progress_bar.set_title("Scanning Source Images")
        self.progress_bar.start_progress(100)

        # Create lists to store the URLs and titles
        image_page_urls = []
        image_names = []

        self._last_page_count = None

        try:
            scanned = None
            try:
                scanned = self._scan_images_via_requests()
            except Exception as e:
                self.reporter.detail(f"fast scan unavailable ({e}), using browser")

            if scanned is None:
                image_names, image_page_urls = self._scan_images_via_selenium()
            else:
                image_names, image_page_urls = scanned

            # Create a pandas DataFrame
            if image_names and image_page_urls:
                images = pd.DataFrame({
                    'Name': image_names,
                    'Image Page': image_page_urls
                })
                success = True
            else:
                images = []
                success = False

        except Exception as e:
            self.reporter.step("Images", "fail", f"{e}")
            images = []
            success = False

        finally:
            self.progress_bar.finish_progress()

        return images, success

    def get_image_url(self, session, image_page_url):
        """
        Given an image page URL, retrieve the image URL.
        """
        try:
            # Make a GET request to the image page URL using the authenticated session
            response = session.get(image_page_url, timeout=REQUEST_TIMEOUT)

            # Convert the webpage to soup
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the div element with id="original_image_container" and style="display:none;"
            image_container = soup.find('div', id='original_image_container', style='display:none;')

            # Returning None rather than printing: this runs once per image on a
            # worker thread, and the caller already aggregates and reports the
            # failures (capped at MAX_REPORTED_FAILURES) instead of flooding.
            if image_container is None:
                return None

            image_element = image_container.find('img')
            if image_element is None:
                return None

            # Resolve against the page URL in case the src is relative
            image_url = image_element.get('src')
            return urljoin(image_page_url, image_url) if image_url else None

        except Exception:
            return None

    def get_image_urls(self, image_page_urls):
        """
        Given a list of image page URLs, retrieve the image URLs for each image page.
        This function uses requests to authenticate with the website and retrieve
        the image URLs, because it is thread-safe, unlike Selenium.
        
        Returns:
            list: A list of image URLs in the same order as the input image_page_urls,
                  with None for any URLs that couldn't be retrieved.
        """
        # List to hold all the image URLs (with same length as image_page_urls)
        image_urls = [None] * len(image_page_urls)
    
        # Initialize progress bar
        self.progress_bar.set_title(f"Retrieving URLs for {len(image_page_urls)} Images")
        self.progress_bar.start_progress(100)
    
        if not image_page_urls:
            self.progress_bar.finish_progress()
            return image_urls

        try:
            # Reuse the shared authenticated session so these requests ride the same
            # warm connection pool as the page scan and the image downloads.
            session = self._get_session()

            with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
                # Submit the image_url retrieval tasks to the thread pool
                # Include the index to maintain order
                future_to_idx_url = {
                    executor.submit(self.get_image_url, session, url): (idx, url)
                    for idx, url in enumerate(image_page_urls)
                }

                # Retrieve the completed results as they become available
                total_urls = len(future_to_idx_url)
                completed = 0
                failures = 0
                last_percent = -1

                for future in concurrent.futures.as_completed(future_to_idx_url):
                    idx, url = future_to_idx_url[future]
                    try:
                        image_url = future.result()

                        # Store result at the correct index
                        image_urls[idx] = image_url

                        if not image_url:
                            failures += 1
                            if failures <= MAX_REPORTED_FAILURES:
                                self.reporter.detail(f"no URL found for {url}")

                    except Exception as e:
                        failures += 1
                        if failures <= MAX_REPORTED_FAILURES:
                            self.reporter.detail(f"{url}: {e}")

                    # Update progress bar only when the whole number percent moves;
                    # update_progress_percentage runs processEvents on every call.
                    completed += 1
                    progress_percent = int(completed / total_urls * 100)
                    if progress_percent != last_percent:
                        self.progress_bar.update_progress_percentage(progress_percent)
                        last_percent = progress_percent

                if failures > MAX_REPORTED_FAILURES:
                    self.reporter.detail(
                        f"...and {failures - MAX_REPORTED_FAILURES} more URL failures")

        except Exception as e:
            raise Exception(f"ERROR: Failed to retrieve image URLs: {str(e)}")
    
        finally:
            self.progress_bar.finish_progress()
    
        return image_urls

    @staticmethod
    def download_image(session, url, path, timeout=REQUEST_TIMEOUT):
        """
        Download an image from a URL and save it to a directory.

        Args:
            session (requests.Session): Shared authenticated session. Reusing it is
                what keeps the connection warm — the previous module-level
                requests.get opened a fresh TCP+TLS connection per image.
            url (str): URL of the image to download
            path (str): Local path where the image should be saved
            timeout (tuple): (connect, read) timeout for the request in seconds

        Returns:
            tuple: (image_path, status, num_bytes, reason)
                - image_path: Path where the image should be saved
                - status: 'downloaded', 'skipped' (already present) or 'failed'
                - num_bytes: Bytes written this call (0 when skipped or failed)
                - reason: Short failure description, or None on success

            Statuses are reported rather than printed so the caller can aggregate
            them into one summary line instead of emitting a warning per image.
        """
        # Do not re-download images that already exist
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path, 'skipped', 0, None

        if not url:
            return path, 'failed', 0, "no source URL"

        try:
            # Send a GET request to the image URL with timeout
            response = session.get(url, timeout=timeout, stream=True)

            # Check if the response was successful
            if response.status_code == 200:
                # Save the image to the specified path. The parent directory is
                # created once by the caller rather than per image.
                written = 0
                with open(path, 'wb') as f:
                    # Use stream mode for large files
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)

                # Verify the file was created and has content
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return path, 'downloaded', written, None

                return path, 'failed', 0, "downloaded file was empty"

            return path, 'failed', 0, f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            return path, 'failed', 0, "timed out"
        except requests.exceptions.ConnectionError:
            return path, 'failed', 0, "connection error"
        except Exception as e:
            return path, 'failed', 0, str(e)

    def download_images(self, dataframe):
        """
        Download images from URLs in a pandas dataframe and save them to a
        directory.

        Returns:
            dict: Counts keyed 'downloaded', 'skipped', 'failed' plus 'bytes'.
        """
        # Save the dataframe of images locally
        csv_file = os.path.join(self.source_dir, "images.csv")
        dataframe.to_csv(csv_file)

        # Check if the CSV file was saved before trying to download
        if not os.path.exists(csv_file):
            raise Exception("ERROR: Unable to save image CSV file")

        # Initialize progress bar
        self.progress_bar.set_title(f"Downloading {len(dataframe)} Images")
        self.progress_bar.start_progress(100)

        # Create the image directory once, rather than once per image
        image_dir = os.path.join(self.source_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        session = self._get_session()
        total = len(dataframe)
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'bytes': 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.download_workers) as executor:
            results = []

            # itertuples/zip over the two columns avoids iterrows(), which builds a
            # fresh Series object for every single row.
            for name, url in zip(dataframe['Name'], dataframe['Image URL']):
                # basename guards against a name containing path separators
                path = os.path.join(image_dir, os.path.basename(str(name)))
                # Add the download task to the executor
                results.append(executor.submit(self.download_image, session, url, path))

            # Wait for all tasks to complete and collect the results
            last_percent = -1

            for idx, result in enumerate(concurrent.futures.as_completed(results)):

                try:
                    path, status, num_bytes, reason = result.result()
                    stats[status] += 1
                    stats['bytes'] += num_bytes

                    if status == 'failed' and stats['failed'] <= MAX_REPORTED_FAILURES:
                        self.reporter.detail(f"{os.path.basename(path)}: {reason}")

                except Exception as e:
                    stats['failed'] += 1
                    if stats['failed'] <= MAX_REPORTED_FAILURES:
                        self.reporter.detail(f"{e}")

                # Update progress bar only on whole-percent changes; each call runs
                # processEvents, so per-image updates throttled the download itself.
                progress_percent = int((idx + 1) / total * 100)
                if progress_percent != last_percent:
                    self.progress_bar.update_progress_percentage(progress_percent)
                    last_percent = progress_percent

        if stats['failed'] > MAX_REPORTED_FAILURES:
            self.reporter.detail(
                f"...and {stats['failed'] - MAX_REPORTED_FAILURES} more failures")

        # Finish the progress bar
        self.progress_bar.finish_progress()

        return stats

    def import_downloaded_data(self):
        """Import previously downloaded images, labelsets, and annotations into the current project."""
        if not self.downloaded_source_dirs:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            image_paths = []
            labelset_paths = []
            annotation_paths = []

            for source_dir in self.downloaded_source_dirs:
                images_dir = os.path.join(source_dir, "images")
                if os.path.isdir(images_dir):
                    for name in os.listdir(images_dir):
                        if os.path.splitext(name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
                            image_paths.append(os.path.join(images_dir, name))

                labelset_path = os.path.join(source_dir, "labelset.csv")
                if os.path.exists(labelset_path):
                    labelset_paths.append(labelset_path)

                annotations_path = os.path.join(source_dir, "annotations.csv")
                if os.path.exists(annotations_path):
                    annotation_paths.append(annotations_path)

            # Import images first so annotations can be matched against loaded images
            if image_paths:
                self.main_window.import_images._process_image_files(image_paths, suppress_errors=True)

                # Ensure there's an active image so annotation import doesn't bail out
                if not self.main_window.annotation_window.active_image:
                    self.main_window.image_window.load_image_by_path(image_paths[0])

            # Import labelsets (labels only; safe to run for each source)
            for labelset_path in labelset_paths:
                self.main_window.import_coralnet_labels.import_coralnet_labels(labelset_path)

            # Import all downloaded annotation files at once
            if annotation_paths:
                self.main_window.import_coralnet_annotations.import_annotations(annotation_paths)

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred while importing downloaded data: {str(e)}")

        finally:
            QApplication.restoreOverrideCursor()
