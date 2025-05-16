import os
import time
import traceback
import ujson as json

import concurrent
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox, QGroupBox,
                             QFormLayout, QApplication, QComboBox, QTextEdit,
                             QFileDialog, QSpinBox)

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


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
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_source_layout(self):
        """Setup the source ID input section."""
        source_group = QGroupBox("CoralNet Source")
        form_layout = QFormLayout()

        # Source ID input
        self.source_id_input = QLineEdit()
        form_layout.addRow("Source ID:", self.source_id_input)

        # Output directory with browse button
        output_dir_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.select_output_dir)
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
        form_layout.addRow("Download Metadata:", self.metadata_dropdown)

        # Dropdown for labelset
        self.labelset_dropdown = QComboBox()
        self.labelset_dropdown.addItems(["True", "False"])
        form_layout.addRow("Download Labelset:", self.labelset_dropdown)

        # Dropdown for annotations
        self.annotations_dropdown = QComboBox()
        self.annotations_dropdown.addItems(["True", "False"])
        form_layout.addRow("Download Annotations:", self.annotations_dropdown)

        # Dropdown for images
        self.images_dropdown = QComboBox()
        self.images_dropdown.addItems(["True", "False"])
        form_layout.addRow("Download Images:", self.images_dropdown)

        # Set the form layout to the group box
        options_group.setLayout(form_layout)

        # Add the group box to the main layout
        self.layout.addWidget(options_group)

    def setup_parameters_layout(self):
        """Setup the parameters section."""
        parameters_group = QGroupBox("Parameters")
        form_layout = QFormLayout()

        # Image fetch rate input
        self.image_fetch_rate_input = QSpinBox()
        self.image_fetch_rate_input.setMinimum(3)
        self.image_fetch_rate_input.setValue(5)
        form_layout.addRow("Image Fetch Rate (sec):", self.image_fetch_rate_input)

        # Image break time input
        self.fetch_break_time_input = QSpinBox()
        self.fetch_break_time_input.setMinimum(3)
        self.fetch_break_time_input.setValue(5)
        form_layout.addRow("Image Fetch Break Time (sec):", self.fetch_break_time_input)

        # Set the form layout to the group box
        parameters_group.setLayout(form_layout)

        # Add the group box to the main layout
        self.layout.addWidget(parameters_group)

    def setup_buttons_layout(self):
        """Setup the download and exit buttons"""
        button_layout = QHBoxLayout()

        # Add debug toggle button with bug icon
        self.debug_button = QPushButton()
        self.debug_button.setIcon(get_icon("www.png"))
        self.debug_button.setToolTip("Toggle Headless Mode")
        self.debug_button.setCheckable(True)
        self.debug_button.setMaximumWidth(30)
        
        # Initialize button state based on headless property
        self.debug_button.setChecked(not self.headless)
        
        # Improved toggle handler that ensures button state matches headless state
        def toggle_headless(checked):
            self.headless = not checked
            # Ensure button state matches headless value
            if self.debug_button.isChecked() != (not self.headless):
                self.debug_button.setChecked(not self.headless)
        
        self.debug_button.toggled.connect(toggle_headless)
        button_layout.addWidget(self.debug_button)

        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self.start_download)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.reject)

        button_layout.addWidget(self.download_button)
        button_layout.addWidget(self.exit_button)

        # Add to main layout
        self.layout.addLayout(button_layout)

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

    # def initialize_driver(self):
    #     """
    #     Check if Chrome browser is installed.
    #     """
    #     success = False

    #     options = Options()
    #     # Silence, please.
    #     options.add_argument("--log-level=3")

    #     if self.headless:
    #         # Add headless argument
    #         options.add_argument('headless')
    #         # Needed to avoid timeouts when running in headless mode
    #         options.add_experimental_option('extensionLoadTimeout', 3600000)

    #     # Modify where the downloads go
    #     prefs = {
    #         "download.default_directory": self.source_dir,
    #         "download.prompt_for_download": False,
    #         "download.directory_upgrade": True,
    #         "safebrowsing.enabled": False,
    #         "profile.managed_default_content_settings.images": 2,
    #         "profile.managed_default_content_settings.stylesheet": 2,
    #         "profile.managed_default_content_settings.fonts": 2,
    #     }
    #     options.add_experimental_option("prefs", prefs)

    #     # Initialize progress bar
    #     self.progress_bar.set_title("Checking for Google Chrome")
    #     self.progress_bar.start_progress(100)

    #     try:
    #         # Check if ChromeDriver path is already in PATH
    #         chrome_driver_path = "chromedriver.exe"  # Adjust the name if needed
    #         if not any(
    #             os.path.exists(os.path.join(directory, chrome_driver_path))
    #             for directory in os.environ["PATH"].split(os.pathsep)
    #         ):
    #             # If it's not in PATH, attempt to install it
    #             chrome_driver_path = ChromeDriverManager().install()

    #             if not chrome_driver_path:
    #                 raise Exception("ERROR: ChromeDriver installation failed.")
    #             else:
    #                 # Add the ChromeDriver directory to the PATH environment variable
    #                 os.environ["PATH"] += os.pathsep + os.path.dirname(chrome_driver_path)

    #         # Attempt to open a browser
    #         self.driver = webdriver.Chrome(options=options)
    #         success = True

    #     except Exception as e:
    #         print(f"WARNING: Google Chrome could not be used\n{str(e)}")

    #     finally:
    #         self.progress_bar.finish_progress()

    #     return success
                    
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
                        self.progress_bar.update_progress(50, f"Trying alternative driver setup: {e}")
                        service = ChromeService(ChromeDriverManager().install())
                        self.driver = webdriver.Chrome(service=service, options=options)
                        success = True
                        
                # Last resort: try finding local chromedriver in PATH
                except Exception as e:
                    self.progress_bar.update_progress(75, f"Trying default driver: {e}")
                    self.driver = webdriver.Chrome(options=options)
                    success = True
                    
            # Handle older Selenium versions as fallback
            except ImportError:
                self.progress_bar.update_progress(85, "Using legacy driver setup")
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

        try:
            for source_id in source_ids:
                self.progress_bar.set_title(f"Downloading Data from Source {source_id}")
                self.source_id = source_id

                # Start the download process for this source ID
                self.download()

            QMessageBox.information(self, "Download Complete", "Download completed successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"{str(e)}")

        finally:
            # Make cursor not busy
            QApplication.restoreOverrideCursor()

            if self.progress_bar:
                self.progress_bar.finish_progress()
                self.progress_bar.close()
                self.progress_bar = None

            if self.driver:
                self.driver.quit()
                self.driver = None

            self.logged_in = False

    def download(self):
        """Run the download process"""
        # Create source directory (normalized path needed for Selenium)
        self.source_dir = os.path.normpath(f"{os.path.abspath(self.output_dir)}\\{str(self.source_id)}")
        os.makedirs(self.source_dir, exist_ok=True)

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
            if not self.download_metadata():
                print("Failed to download metadata (see console log)")

        # Download labelset if selected
        if self.download_options.get('labelset', False):
            if not self.download_labelset():
                print("Failed to download labelset (see console log)")

        # Download annotations if selected
        if self.download_options.get('annotations', False):
            if not self.download_annotations():
                print("Failed to download annotations (see console log)")

        # Download images if selected
        if self.download_options.get('images', False):
            images, success = self.get_images()

            if not success:
                raise Exception("Failed while scanning for images (see console log)")

            if len(images):
                # Get image URLs for each of the images
                images['Image URL'] = self.get_image_urls(images['Image Page'].tolist())
                # Download images
                self.download_images(images)

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

            # Find the login button
            path = "//input[@type='submit'][@value='Sign in']"
            login_button = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, path)))

            # Enter the username and password
            username_input.send_keys(self.driver.capabilities['credentials']['username'])
            password_input.send_keys(self.driver.capabilities['credentials']['password'])

            # Click the login button
            time.sleep(3)
            login_button.click()

            # Confirm login was successful; after 10 seconds, throw an error.
            path = "//button[text()='Sign out']"

            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, path)))

            # Login was successful
            success = True
            self.logged_in = True

        except Exception as e:
            print(f"ERROR: Could not login with {username}\n{str(e)}")

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
                meta.to_csv(f"{self.source_dir}\\metadata.csv")

                # Check that it was saved
                if os.path.exists(f"{self.source_dir}\\metadata.csv"):
                    print("Metadata saved successfully")
                    success = True
                else:
                    raise Exception("Metadata could not be saved")

        except Exception as e:
            print(f"ERROR: Issue with downloading metadata: {str(e)}")

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
                labelset.to_csv(f"{self.source_dir}\\labelset.csv")

                # Check that it was saved
                if os.path.exists(f"{self.source_dir}\\labelset.csv"):
                    print("Labelset saved successfully")
                    success = True
                else:
                    raise Exception("Labelset could not be saved")

        except Exception as e:
            print(f"ERROR: Issue with downloading labelset: {str(e)}")

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

            while "Working" in go_button.accessible_name:
                time.sleep(3)

            # Wait for the download to complete
            time.sleep(10)
            
            # Check that it was saved
            if os.path.exists(f"{self.source_dir}\\annotations.csv"):
                print("Annotations saved successfully")
                success = True
            else:
                raise Exception("Annotations may not have been saved")

        except Exception as e:
            print(f"ERROR: Issue with downloading annotations: {str(e)}")

        finally:
            self.progress_bar.finish_progress()

        return success

    def get_images(self):
        """
        Given a source ID, retrieve the image names and page URLs.
        Returns a DataFrame containing image names and their page URLs.
        """
        # Initialize result variables
        images = []
        success = False

        # Initialize progress bar
        self.progress_bar.set_title("Accessing Source Images")
        self.progress_bar.start_progress(100)

        try:
            # Go to the images page
            self.driver.get(self.authentication_dialog.CORALNET_URL + f"/source/{self.source_id}/browse/images/")
            # Get the page source HTML, and the total number of pages
            page_element = self.driver.find_element(By.CSS_SELECTOR, 'div.line')
            total_pages = int(page_element.text.split(" ")[-1]) // 20 + 1
            print(f"Found {total_pages} pages of images")

        except Exception:
            raise Exception("Could not determine total amount of images; please report this issue")

        finally:
            # Update progress bar
            self.progress_bar.finish_progress()

        # Initialize progress bar
        self.progress_bar.set_title("Scanning Source Images")
        self.progress_bar.start_progress(100)

        # Create lists to store the URLs and titles
        image_page_urls = []
        image_names = []

        try:
            current_page = 1
            has_next_page = True

            # Loop through all pages
            while has_next_page and current_page <= total_pages:

                # Let page elements fully load
                time.sleep(self.image_fetch_rate)

                try:
                    # Find all the image elements
                    url_elements = self.driver.find_elements(By.CSS_SELECTOR, '.thumb_wrapper a')
                    name_elements = self.driver.find_elements(By.CSS_SELECTOR, '.thumb_wrapper img')
                except Exception as e:
                    print(f"Warning: Fetching too fast, taking a {self.fetch_break_time} second break")
                    time.sleep(self.fetch_break_time)
                    continue

                # Iterate over the image elements
                for url_element, name_element in zip(url_elements, name_elements):
                    # Extract the href attribute (URL)
                    image_page_url = url_element.get_attribute('href')
                    image_page_urls.append(image_page_url)

                    # Extract the title attribute (image name)
                    image_name = name_element.get_attribute('alt')
                    image_names.append(image_name)

                try:
                    # Check if there is a next page button and it's enabled
                    element_text = 'form.no-padding [type="submit"][value=">"]'

                    try:
                        next_button = self.driver.find_element(By.CSS_SELECTOR, element_text)
                        button_exists = True
                    except:
                        # Element not found, no more pages
                        button_exists = False
                        has_next_page = False

                    if button_exists and next_button.is_displayed() and next_button.is_enabled():
                        # Store current page identifier to verify page change
                        current_page_identifier = self.driver.find_element(By.CSS_SELECTOR, '.line')

                        if not current_page_identifier.text:
                            raise Exception("Could not determine current page number")

                        # Click the next button
                        next_button.click()
                        # Increase page count
                        current_page += 1

                except Exception as e:
                    print(f"Error navigating to next page: {str(e)}")
                    has_next_page = False

                # Update progress bar given total_pages
                progress_percent = int((current_page / total_pages) * 100)
                self.progress_bar.update_progress_percentage(progress_percent)

            # Create a pandas DataFrame
            if image_names and image_page_urls:
                images = pd.DataFrame({
                    'Name': image_names,
                    'Image Page': image_page_urls
                })
                print(f"Retrieved {len(images)} images from source {self.source_id}")
                success = True
            else:
                print(f"No images found for source {self.source_id}")
                images = []
                success = False

        except Exception as e:
            print(f"ERROR: Issue retrieving images: {str(e)}")
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
            response = session.get(image_page_url)
            cookies = response.cookies

            # Convert the webpage to soup
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the div element with id="original_image_container" and style="display:none;"
            image_container = soup.find('div', id='original_image_container', style='display:none;')

            # Find the img element within the div and get the toolbox attribute
            image_url = image_container.find('img').get('src')

            return image_url

        except Exception as e:
            print(f"ERROR: Unable to get image URL from image page: {e}")
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
    
        try:
            # Send a GET request to the login page to retrieve the login form
            response = requests.get(self.authentication_dialog.LOGIN_URL, timeout=30)
    
            # Pass along the cookies
            cookies = response.cookies
    
            # Parse the HTML of the response using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
    
            # Extract the CSRF token from the HTML of the login page
            csrf_token = soup.find("input", attrs={"name": "csrfmiddlewaretoken"})
    
            # Create a dictionary with the login form fields and their values
            data = {
                "username": self.username,
                "password": self.password,
                "csrfmiddlewaretoken": csrf_token["value"],
            }
    
            # Include the "Referer" header in the request
            headers = {
                "Referer": self.authentication_dialog.LOGIN_URL,
            }
    
            # Use requests.Session to create a session that will maintain your login state
            session = requests.Session()
    
            # Use session.post() to submit the login form
            session.post(self.authentication_dialog.LOGIN_URL, data=data, headers=headers, cookies=cookies)
    
            # Use a thread pool with a reasonable number of workers
            with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                # Submit the image_url retrieval tasks to the thread pool
                # Include the index to maintain order
                future_to_idx_url = {
                    executor.submit(self.get_image_url, session, url): (idx, url)
                    for idx, url in enumerate(image_page_urls)
                }
    
                # Retrieve the completed results as they become available
                total_urls = len(future_to_idx_url)
                completed = 0
                
                for future in concurrent.futures.as_completed(future_to_idx_url):
                    idx, url = future_to_idx_url[future]
                    try:
                        image_url = future.result()
                        
                        # Store result at the correct index
                        image_urls[idx] = image_url
                        
                        if not image_url:
                            print(f"Warning: Failed to retrieve image URL for {url}")
    
                    except Exception as e:
                        print(f"ERROR: Failed to retrieve URL for {url}: {str(e)}")
                    
                    # Update progress bar
                    completed += 1
                    progress_percent = int(completed / total_urls * 100)
                    self.progress_bar.update_progress_percentage(progress_percent)
    
        except Exception as e:
            raise Exception(f"ERROR: Failed to retrieve image URLs: {str(e)}")
    
        finally:
            self.progress_bar.finish_progress()
    
        return image_urls

    @staticmethod
    def download_image(url, path, timeout=30):
        """
        Download an image from a URL and save it to a directory.

        Args:
            url (str): URL of the image to download
            path (str): Local path where the image should be saved
            timeout (int): Timeout for the request in seconds

        Returns:
            tuple: (image_path, success_flag)
                - image_path: Path where the image should be saved
                - success_flag: True if download was successful, False otherwise
        """
        # Do not re-download images that already exist
        if os.path.exists(path):
            return path, True

        try:
            # Send a GET request to the image URL with timeout
            response = requests.get(url, timeout=timeout, stream=True)

            # Check if the response was successful
            if response.status_code == 200:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)

                # Save the image to the specified path
                with open(path, 'wb') as f:
                    # Use stream mode for large files
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Verify the file was created and has content
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return path, True
                else:
                    print(f"Warning: Downloaded file {path} is empty or doesn't exist")
                    return path, False
            else:
                print(f"Warning: Failed to download {url}, status code: {response.status_code}")
                return path, False

        except requests.exceptions.Timeout:
            print(f"Warning: Timeout while downloading {url}")
            return path, False
        except requests.exceptions.ConnectionError:
            print(f"Warning: Connection error while downloading {url}")
            return path, False
        except Exception as e:
            print(f"Warning: Failed to download {url} - {str(e)}")
            return path, False

    def download_images(self, dataframe):
        """
        Download images from URLs in a pandas dataframe and save them to a
        directory.
        """
        # Save the dataframe of images locally
        csv_file = f"{self.source_dir}\\images.csv"
        dataframe.to_csv(csv_file)

        # Check if the CSV file was saved before trying to download
        if os.path.exists(csv_file):
            print("Saved image dataframe as CSV file")
        else:
            raise Exception("ERROR: Unable to save image CSV file")

        # Initialize progress bar
        self.progress_bar.set_title(f"Downloading {len(dataframe)} Images")
        self.progress_bar.start_progress(100)

        # Create the image directory if it doesn't exist (it should)
        image_dir = f"{self.source_dir}/images/"

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            results = []

            for index, row in dataframe.iterrows():
                # Get the image name and URL from the dataframe
                name = row['Name']
                url = row['Image URL']
                path = image_dir + name
                # Add the download task to the executor
                results.append(executor.submit(self.download_image, url, path))

            # Wait for all tasks to complete and collect the results
            for idx, result in enumerate(concurrent.futures.as_completed(results)):

                try:
                    # Get the downloaded image path
                    downloaded_image_path, downloaded = result.result()

                    if not downloaded:
                        raise Exception(f"Failed to download image {os.path.basename(downloaded_image_path)}")

                except Exception as e:
                    print(f"ERROR: {str(e)}")

                # Update progress bar
                progress_percent = int((idx + 1) / len(dataframe) * 100)
                self.progress_bar.update_progress_percentage(progress_percent)

        # Finish the progress bar
        self.progress_bar.finish_progress()
