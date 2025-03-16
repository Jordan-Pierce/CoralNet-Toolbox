import os
import json
import requests
from bs4 import BeautifulSoup
import traceback
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QGroupBox, 
                             QFormLayout, QApplication, QComboBox, QTextEdit,
                             QFileDialog)

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ----------------------------------------------------------------------------------------------------------------------
# Worker Thread Class
# ----------------------------------------------------------------------------------------------------------------------


class DownloadWorker(QThread):
    """Worker thread for downloading data from CoralNet"""
    
    # Signal to update the status text
    update_status = pyqtSignal(str)
    # Signal to indicate download completion
    download_complete = pyqtSignal(bool, str)
    
    def __init__(self, dialog, source_id, output_dir, download_options, auth_token=None, username=None, password=None):
        super(DownloadWorker, self).__init__()
        self.download_dialog = dialog  # Reference to the parent dialog
        self.source_id = source_id
        self.output_dir = output_dir
        self.download_options = download_options
        self.auth_token = auth_token
        self.username = username
        self.password = password
        
    def run(self):
        """Run the download process in a separate thread"""
        try:
            self.update_status.emit(f"Initializing download for Source ID: {self.source_id}...")
            
            # Check which browsers are available
            browser_name = self.download_dialog.check_for_browsers()
            if not browser_name:
                self.update_status.emit("ERROR: No supported browser found. Please install Chrome or Firefox.")
                self.download_complete.emit(False, "No supported browser found")
                return
                
            # Create source directory
            source_dir = f"{os.path.abspath(self.output_dir)}/{str(self.source_id)}/"
            os.makedirs(source_dir, exist_ok=True)
            
            # Login to CoralNet
            self.update_status.emit(f"Logging in to CoralNet...")
            driver, success = self.download_dialog.login(browser_name)
            
            # Download based on selected options
            results = {}
            
            # Download metadata if selected
            if self.download_options.get('metadata', False):
                self.update_status.emit(f"Downloading metadata for Source ID: {self.source_id}...")
                try:
                    driver, metadata = self.download_dialog.download_metadata(driver, self.source_id, source_dir)
                    results['metadata'] = "✓ Successfully downloaded metadata"
                except Exception as e:
                    results['metadata'] = f"✗ Failed to download metadata: {str(e)}"
                    self.update_status.emit(results['metadata'])
            
            # Download labelset if selected
            if self.download_options.get('labelset', False):
                self.update_status.emit(f"Downloading labelset for Source ID: {self.source_id}...")
                try:
                    driver, labelset = self.download_dialog.download_labelset(driver, self.source_id, source_dir)
                    results['labelset'] = "✓ Successfully downloaded labelset"
                except Exception as e:
                    results['labelset'] = f"✗ Failed to download labelset: {str(e)}"
                    self.update_status.emit(results['labelset'])
            
            # Download annotations if selected
            if self.download_options.get('annotations', False):
                self.update_status.emit(f"Downloading annotations for Source ID: {self.source_id}...")
                try:
                    driver, annotations = self.download_dialog.download_annotations(driver, self.source_id, source_dir)
                    results['annotations'] = "✓ Successfully downloaded annotations"
                except Exception as e:
                    results['annotations'] = f"✗ Failed to download annotations: {str(e)}"
                    self.update_status.emit(results['annotations'])
            
            # Download images if selected
            if self.download_options.get('images', False):
                self.update_status.emit(f"Getting image information for Source ID: {self.source_id}...")
                try:
                    # Get images metadata
                    driver, images = self.download_dialog.get_images(driver, self.source_id)
                    
                    if images is not None and not images.empty:
                        # Get image URLs
                        self.update_status.emit(f"Retrieving image URLs...")
                        image_pages = images['Image Page'].tolist()
                        driver, images['Image URL'] = self.download_dialog.get_image_urls(driver, image_pages)
                        
                        # Download images
                        self.update_status.emit(f"Downloading {len(images)} images...")
                        self.download_dialog.download_images(images, source_dir)
                        results['images'] = f"✓ Successfully downloaded {len(images)} images"
                    else:
                        results['images'] = "✗ No images found for this source"
                except Exception as e:
                    results['images'] = f"✗ Failed to download images: {str(e)}"
                    self.update_status.emit(results['images'])
            
            # Close the driver
            if driver:
                driver.quit()
                
            # Summarize results
            summary = f"Download Summary for Source ID {self.source_id}:\n\n"
            for key, value in results.items():
                summary += f"{value}\n"
                
            self.update_status.emit(summary)
            self.download_complete.emit(True, source_dir)
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            self.update_status.emit(error_msg)
            self.download_complete.emit(False, error_msg)
            
            # Ensure driver is closed on exception
            try:
                if 'driver' in locals() and driver:
                    driver.quit()
            except:
                pass


# ----------------------------------------------------------------------------------------------------------------------
# Download Dialog
# ----------------------------------------------------------------------------------------------------------------------


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
        
        # Initialize download worker to None
        self.download_worker = None
        
        self.setWindowTitle("Download from CoralNet")
        self.resize(600, 500)  # Width, height
        
        # Create the layout
        self.layout = QVBoxLayout(self)
        
        # Setup the info layout
        self.setup_info_layout()
        # Setup the source layout
        self.setup_source_layout()
        # Setup the options layout
        self.setup_options_layout()
        # Setup the status layout
        self.setup_status_layout()
        # Setup buttons layout
        self.setup_buttons_layout()
        
    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        info_label = QLabel(
            "Download data from a CoralNet source. Specify the source ID and select which items to download."
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
        
        # Dropdown for images
        self.images_dropdown = QComboBox()
        self.images_dropdown.addItems(["True", "False"])
        form_layout.addRow("Download Images:", self.images_dropdown)
        
        # Dropdown for labelset
        self.labelset_dropdown = QComboBox()
        self.labelset_dropdown.addItems(["True", "False"])
        form_layout.addRow("Download Labelset:", self.labelset_dropdown)
        
        # Dropdown for annotations
        self.annotations_dropdown = QComboBox()
        self.annotations_dropdown.addItems(["True", "False"])
        form_layout.addRow("Download Annotations:", self.annotations_dropdown)
        
        # Dropdown for metadata
        self.metadata_dropdown = QComboBox()
        self.metadata_dropdown.addItems(["True", "False"])
        form_layout.addRow("Download Metadata:", self.metadata_dropdown)
        
        # Set the form layout to the group box
        options_group.setLayout(form_layout)
        
        # Add the group box to the main layout
        self.layout.addWidget(options_group)
    
    def setup_status_layout(self):
        """Setup the status display section"""
        status_group = QGroupBox("Download Status")
        status_layout = QVBoxLayout()
        
        # Status text area
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Download status will appear here...")
        status_layout.addWidget(self.status_text)
        
        status_group.setLayout(status_layout)
        self.layout.addWidget(status_group)
    
    def setup_buttons_layout(self):
        """Setup the download and exit buttons"""
        button_layout = QHBoxLayout()
        
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
        source_id = self.source_id_input.text().strip()
        if not source_id:
            QMessageBox.warning(self, "Input Error", "Source ID is required.")
            return False
        
        try:
            int(source_id)  # Check if it's a valid number
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Source ID must be a number.")
            return False
        
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Input Error", "Output directory is required.")
            return False
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Directory Error", f"Could not create output directory: {str(e)}")
            return False
        
        options = self.get_download_options()
        if not any(options.values()):
            QMessageBox.warning(self, "Input Error", "Please select at least one download option.")
            return False
        
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
        source_id = self.source_id_input.text().strip()
        output_dir = self.output_dir_input.text().strip()
        download_options = self.get_download_options()
        
        # Get credentials from auth dialog
        auth_token = self.authentication_dialog.get_auth_token()
        username = self.authentication_dialog.username_input.text()
        password = self.authentication_dialog.password_input.text()
        
        # Clear status text
        self.status_text.clear()
        self.status_text.append(f"Starting download for Source ID: {source_id}...")
        
        # Disable download button during the process
        self.download_button.setEnabled(False)
        
        # Create and start the download worker thread
        self.download_worker = DownloadWorker(
            self,  # Pass reference to this dialog 
            source_id, 
            output_dir, 
            download_options,
            auth_token,
            username,
            password
        )
        
        # Connect signals
        self.download_worker.update_status.connect(self.update_status)
        self.download_worker.download_complete.connect(self.on_download_complete)
        
        # Start the worker
        self.download_worker.start()
    
    def update_status(self, message):
        """Update the status text area with a message"""
        self.status_text.append(message)
        # Scroll to the bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )
    
    def on_download_complete(self, success, message):
        """Handle download completion"""
        self.download_button.setEnabled(True)
        
        if success:
            self.update_status("\nDownload completed successfully!")
            QMessageBox.information(
                self,
                "Download Complete",
                f"Download completed successfully.\nFiles saved to: {message}"
            )
        else:
            self.update_status("\nDownload failed!")
            QMessageBox.critical(
                self,
                "Download Failed",
                f"Download failed: {message}"
            )

    def download_metadata(self, driver, source_id, source_dir=None):
        """
        Given a source ID, download the labelset.
        """
        # To hold the metadata
        meta = []

        # Go to the meta page
        driver.get(self.authentication_dialog.CORALNET_URL + f"/source/{source_id}/")

        # First check that this is existing source the user has access to
        try:
            # Check the permissions
            driver, status = self.check_permissions(driver)

            # Check the status
            if "Page could not be found" in status.text or "don't have permission" in status.text:
                raise Exception(status.text.split('.')[0])

        except Exception as e:
            raise Exception(f"ERROR: {e} or you do not have permission to access it")

        self.update_status(f"Downloading model metadata for {source_id}")

        try:
            # Convert the page to soup
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            script = None

            # Of the scripts, find the one containing model metadata
            for script in soup.find_all("script"):
                if "Classifier overview" in script.text:
                    script = script.text
                    break

            # If the page doesn't have model metadata, then return None early
            if not script:
                raise Exception("NOTE: No model metadata found")

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

            # Save if user provided an output directory
            if source_dir:
                # Just in case
                os.makedirs(source_dir, exist_ok=True)
                # Save the meta as a CSV file
                meta.to_csv(f"{source_dir}{source_id}_metadata.csv")
                # Check that it was saved
                if os.path.exists(f"{source_dir}{source_id}_metadata.csv"):
                    self.update_status("Metadata saved successfully")
                else:
                    raise Exception("WARNING: Metadata could not be saved")

        except Exception as e:
            self.update_status(f"ERROR: Issue with downloading metadata: {str(e)}")
            meta = None

        return driver, meta

    def download_labelset(self, driver, source_id, source_dir=None):
        """
        Given a source ID, download the labelset.
        """
        # To hold the labelset
        labelset = None

        # Go to the images page
        driver.get(self.authentication_dialog.CORALNET_URL + f"/source/{source_id}/labelset/")

        # First check that this is existing source the user has access to
        try:
            # Check the permissions
            driver, status = self.check_permissions(driver)

            # Check the status
            if "Page could not be found" in status.text or "don't have permission" in status.text:
                raise Exception(status.text.split('.')[0])

        except Exception as e:
            raise Exception(f"ERROR: {e} or you do not have permission to access it")

        self.update_status(f"Downloading labelset for {source_id}")

        try:
            # Get the page source HTML
            html_content = driver.page_source
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the table with id 'label-table'
            table = soup.find('table', {'id': 'label-table'})
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

            # Save if user provided an output directory
            if source_dir:
                # Just in case
                os.makedirs(source_dir, exist_ok=True)
                # Save the labelset as a CSV file
                labelset.to_csv(f"{source_dir}{source_id}_labelset.csv")
                # Check that it was saved
                if os.path.exists(f"{source_dir}{source_id}_labelset.csv"):
                    self.update_status("Labelset saved successfully")
                else:
                    raise Exception("WARNING: Labelset could not be saved")

        except Exception as e:
            self.update_status(f"ERROR: Issue with downloading labelset: {str(e)}")
            labelset = None

        return driver, labelset

    def download_image(self, image_url, image_path):
        """
        Download an image from a URL and save it to a directory. Return the path
        to the downloaded image if download was successful, otherwise return None.
        """
        # Do not re-download images that already exist
        if os.path.exists(image_path):
            return image_path, True

        # Send a GET request to the image URL
        response = requests.get(image_url)

        # Check if the response was successful
        if response.status_code == 200:
            # Save the image to the specified path
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return image_path, True
        else:
            return image_path, False

    def download_images(self, dataframe, source_dir):
        """
        Download images from URLs in a pandas dataframe and save them to a
        directory.
        """
        # Extract source id from path
        source_id = os.path.basename(os.path.normpath(source_dir))
        # Save the dataframe of images locally
        csv_file = f"{source_dir}{source_id}_images.csv"
        dataframe.to_csv(csv_file)
        # Check if the CSV file was saved before trying to download
        if os.path.exists(csv_file):
            self.update_status("Saved image dataframe as CSV file")
        else:
            raise Exception("ERROR: Unable to save image CSV file")

        # Create the image directory if it doesn't exist (it should)
        image_dir = f"{source_dir}/images/"
        os.makedirs(image_dir, exist_ok=True)

        self.update_status(f"Downloading {len(dataframe)} images")

        # To hold the expired images
        expired_images = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
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
                # Get the downloaded image path
                downloaded_image_path, downloaded = result.result()
                # Get the image name from the downloaded image path
                basename = os.path.basename(downloaded_image_path)
                if not downloaded:
                    expired_images.append(basename)
                # Update progress every 10 images
                if idx % 10 == 0:
                    self.update_status(f"Downloaded {idx+1}/{len(dataframe)} images")

        if expired_images:
            self.update_status(f"{len(expired_images)} images had expired before being downloaded")
            self.update_status(f"Saving list of expired images to {source_dir} expired_images.csv")
            expired_images = pd.DataFrame(expired_images, columns=['image_path'])
            expired_images.to_csv(f"{source_dir}{source_id}_expired_images.csv")

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
            image_url = image_container.find('img').get('toolbox')

            return image_url

        except Exception as e:
            self.update_status(f"ERROR: Unable to get image URL from image page: {e}")
            return None

    def get_image_urls(self, driver, image_page_urls):
        """
        Given a list of image page URLs, retrieve the image URLs for each image page.
        This function uses requests to authenticate with the website and retrieve
        the image URLs, because it is thread-safe, unlike Selenium.
        """
        self.update_status("Retrieving image URLs")
        
        # CoralNet login URL
        LOGIN_URL = self.authentication_dialog.LOGIN_URL

        # List to hold all the image URLs
        image_urls = []

        try:
            # Send a GET request to the login page to retrieve the login form
            response = requests.get(LOGIN_URL, timeout=30)

        except Exception as e:
            raise Exception(f"ERROR: CoralNet timed out after 30 seconds.\n{e}")

        # Pass along the cookies
        cookies = response.cookies

        # Parse the HTML of the response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the CSRF token from the HTML of the login page
        csrf_token = soup.find("input", attrs={"name": "csrfmiddlewaretoken"})

        # Create a dictionary with the login form fields and their values
        data = {
            "username": driver.capabilities['credentials']['username'],
            "password": driver.capabilities['credentials']['password'],
            "csrfmiddlewaretoken": csrf_token["value"],
        }

        # Include the "Referer" header in the request
        headers = {
            "Referer": LOGIN_URL,
        }

        # Use requests.Session to create a session that will maintain your login state
        session = requests.Session()

        # Use session.post() to submit the login form
        session.post(LOGIN_URL, data=data, headers=headers, cookies=cookies)

        with ThreadPoolExecutor() as executor:
            # Submit the image_url retrieval tasks to the thread pool
            future_to_url = {executor.submit(self.get_image_url, session, url): url for url in image_page_urls}

            # Retrieve the completed results as they become available
            count = 0
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    image_url = future.result()
                    if image_url:
                        image_urls.append(image_url)
                    count += 1
                    # Update status every 10 images
                    if count % 10 == 0:
                        self.update_status(f"Retrieved {count}/{len(image_page_urls)} image URLs")
                except Exception as e:
                    self.update_status(f"ERROR: issue retrieving image URL for {url}\n{e}")

        self.update_status(f"Retrieved {len(image_urls)} image URLs for {len(image_page_urls)} images")

        return driver, image_urls

    def get_images(self, driver, source_id, prefix="", image_list=None):
        """
        Given a source ID, retrieve the image names, and page URLs.
        """
        from selenium.webdriver.common.by import By
        
        # Go to the images page
        driver.get(self.authentication_dialog.CORALNET_URL + f"/source/{source_id}/browse/images/")

        # First check that this is existing source the user has access to
        try:
            # Check the permissions
            driver, status = self.check_permissions(driver)

            # Check the status
            if "Page could not be found" in status.text or "don't have permission" in status.text:
                raise Exception(status.text.split('.')[0])

        except Exception as e:
            raise Exception(f"ERROR: {e} or you do not have permission to access it")

        # If provided, will limit the search space on CoralNet
        if prefix != "":
            self.update_status(f"Filtering search space using '{prefix}'")
            input_element = driver.find_element(By.CSS_SELECTOR, "#id_image_name")
            input_element.clear()
            input_element.send_keys(prefix)

            # Click submit
            submit_button = driver.find_element(By.CSS_SELECTOR, ".submit_button_wrapper_center input[type='submit']")
            submit_button.click()

        self.update_status(f"Crawling all pages for source {source_id}")

        # Create lists to store the URLs and titles
        image_page_urls = []
        image_names = []

        try:
            # Find the element with the page number
            page_element = driver.find_element(By.CSS_SELECTOR, 'div.line')
            num_pages = int(page_element.text.split(" ")[-1]) // 20 + 1
            page_num = 0

            while True:
                # Find all the image elements
                url_elements = driver.find_elements(By.CSS_SELECTOR, '.thumb_wrapper a')
                name_elements = driver.find_elements(By.CSS_SELECTOR, '.thumb_wrapper img')

                # Iterate over the image elements
                for url_element, name_element in list(zip(url_elements, name_elements)):
                    # Extract the href attribute (URL)
                    image_page_url = url_element.get_attribute('href')
                    image_page_urls.append(image_page_url)

                    # Extract the title attribute (image name)
                    image_name = name_element.get_attribute('title')
                    image_names.append(image_name)

                # Check if there is a next page
                next_button = driver.find_element(By.CSS_SELECTOR, 'a.next')
                if next_button and page_num < num_pages:
                    next_button.click()
                    page_num += 1
                else:
                    break

            # Create a pandas DataFrame
            images = pd.DataFrame({
                'Name': image_names,
                'Image Page': image_page_urls
            })

        except Exception as e:
            self.update_status(f"ERROR: Issue with retrieving images: {str(e)}")
            images = None

        return driver, images

    def check_permissions(self, driver):
        """
        Check the permissions of the current page.
        """
        # Convert the page to soup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find the status element
        status = soup.find('div', {'class': 'status'})

        return driver, status

    def check_for_browsers(self, headless=True):
        """
        Check if Chrome browser is installed.
        """
        self.update_status("Checking for available browsers...")
        
        options = Options()
        # Silence, please.
        options.add_argument("--log-level=3")

        if headless:
            # Add headless argument
            options.add_argument('headless')
            # Needed to avoid timeouts when running in headless mode
            options.add_experimental_option('extensionLoadTimeout', 3600000)

        try:
            # Check if ChromeDriver path is already in PATH
            chrome_driver_path = "chromedriver.exe"  # Adjust the name if needed
            if not any(
                os.path.exists(os.path.join(directory, chrome_driver_path))
                for directory in os.environ["PATH"].split(os.pathsep)
            ):
                # If it's not in PATH, attempt to install it
                chrome_driver_path = ChromeDriverManager().install()

                if not chrome_driver_path:
                    raise Exception("ERROR: ChromeDriver installation failed.")
                else:
                    # Add the ChromeDriver directory to the PATH environment variable
                    os.environ["PATH"] += os.pathsep + os.path.dirname(chrome_driver_path)
                    self.update_status("NOTE: ChromeDriver added to PATH")

            # Attempt to open a browser
            browser = webdriver.Chrome(options=options)

            self.update_status("NOTE: Using Google Chrome")
            return browser

        except Exception as e:
            self.update_status(f"WARNING: Google Chrome could not be used\n{str(e)}")

        raise Exception("ERROR: Issue with getting browser. Exiting")

    def login(self, driver):
        """
        Log in to CoralNet using Selenium.
        """
        self.update_status("Logging in to CoralNet...")

        # Create a variable for success
        success = False

        # Get auth info from the authentication dialog
        username = self.authentication_dialog.username_input.text()
        password = self.authentication_dialog.password_input.text()
        
        # Add credentials to driver capabilities for later use
        driver.capabilities['credentials'] = {
            'username': username,
            'password': password
        }

        # Navigate to the page to login
        driver.get(self.authentication_dialog.CORALNET_URL + "/accounts/login/")

        # Find the username button
        path = "id_username"
        username_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Find the password button
        path = "id_password"
        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Find the login button
        path = "//input[@type='submit'][@value='Sign in']"
        login_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, path)))

        # Enter the username and password
        username_input.send_keys(driver.capabilities['credentials']['username'])
        password_input.send_keys(driver.capabilities['credentials']['password'])

        # Click the login button
        time.sleep(3)
        login_button.click()

        # Confirm login was successful; after 10 seconds, throw an error.
        try:
            path = "//button[text()='Sign out']"

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, path)))

            # Login was successful
            success = True

            self.update_status(f"NOTE: Successfully logged in as {driver.capabilities['credentials']['username']}")

        except Exception as e:
            raise ValueError(f"ERROR: Could not login with {driver.capabilities['credentials']['username']}\n{str(e)}")

        return driver, success
