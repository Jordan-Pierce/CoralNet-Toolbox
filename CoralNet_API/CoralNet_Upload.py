import os
import glob
import sys
import time
import argparse
import subprocess

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from CoralNet import *

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def check_for_browsers(options):
    """
    Check if Chrome, Firefox, and Edge browsers are installed.
    """

    # Greedy search for Chrome, Firefox, and Edge browsers
    try:
        browser = webdriver.Chrome(options=options)
        print("NOTE: Using Google Chrome")
        return browser

    except Exception as e:
        print("Warning: Google Chrome is not installed")

    try:
        browser = webdriver.Edge(options=options)
        print("NOTE: Using Microsoft Edge")
        return browser

    except Exception as e:
        print("Warning: Microsoft Edge is not installed")

    try:
        browser = webdriver.Firefox(options=options)
        print("NOTE: Using Mozilla Firefox")
        return browser

    except Exception as e:
        print("Warning: Firefox is not installed")

    print("ERROR: No browsers are installed. Exiting")
    sys.exit(1)


def login(username, password, driver):
    """
    Log in to CoralNet.
    """

    # Create a variable for success
    success = False

    # Navigate to the page to login
    driver.get(CORALNET_URL + "/accounts/login/")

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
    username_input.send_keys(username)
    password_input.send_keys(password)

    # Click the login button
    time.sleep(3)
    login_button.click()

    # Confirm login was successful; after 60 seconds, throw an error.
    try:
        path = "//a[@href='/accounts/logout/']/span[text()='Sign out']"
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, path)))

        # Login was successful
        success = True

    except Exception as e:
        raise ValueError(f"ERROR: Could not login with {username}\n{e}")

    return driver, success


def check_permissions(driver):
    """
    Check if the user has permission to access the page.
    """

    try:
        # First check that this is existing source the user has access to
        path = "content-container"
        status = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Check the status
        if not status.text:
            raise Exception(f"ERROR: Unable to access page information")

    except Exception as e:
        print(f"ERROR: {e} Exiting.")
        sys.exit(1)

    return driver, status


def upload_images(driver, source_id, images):
    """
    Upload images to CoralNet.
    """

    print("\nNavigating to image upload page...")

    # Variable for success
    success = False

    # Go to the upload page
    driver.get(CORALNET_URL + f"/source/{source_id}/upload/images/")

    # First check that this is existing source the user has access to
    try:
        # Check the permissions
        driver, status = check_permissions(driver)

        # Check the status
        if "Page could not be found" in status.text:
            raise Exception(f"ERROR: {status.text.split('.')[0]}")

    except Exception as e:
        print(f"ERROR: {e} or you do not have permission to access it")
        return driver, success

    # Send the files to CoralNet for upload
    try:
        # Locate the file input field
        path = "//input[@type='file'][@name='files']"
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, path)))

        # Check if the file input field is enabled
        if file_input.is_enabled():
            print(f"NOTE: File input field is enabled")
            for image_path in images:
                print(f"NOTE: Sending {os.path.basename(image_path)}")
                file_input.send_keys(image_path)
        else:
            # File input field is not enabled, something is wrong
            raise ValueError(
                "ERROR: File input field is not enabled; exiting.")

    except Exception as e:
        print(f"ERROR: Could not submit files for upload\n{e}")
        return success

    # Attempt to upload the files to CoralNet
    try:
        # Check if files can be uploaded
        path = "status_display"
        status = WebDriverWait(driver, len(images)).until(
            EC.presence_of_element_located((By.ID, path)))

        # If there are many files, they will be checked
        while "Checking files..." in status.text:
            continue

        # Give the upload status time to update
        time.sleep(3)

        # Get the pre-upload status
        path = "pre_upload_summary"
        pre_status = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Print the pre-upload status
        print(f"\n{pre_status.text}\n")

        # Images can be uploaded
        if "Ready for upload" in status.text:

            # Wait for the upload button to appear
            path = "//button[@id='id_upload_submit']"
            upload_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, path)))

            # Check if the upload button is enabled
            if upload_button.is_enabled():
                # Click the upload button
                upload_button.click()

                # Print the status
                print(f"NOTE: {status.text}")

                # Give the upload status time to update
                time.sleep(3)

                # Get the mid-upload status
                path = "mid_upload_summary"
                mid_status = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, path)))

                while "Uploading..." in status.text:
                    new_upload_status_text = mid_status.text
                    if new_upload_status_text != mid_status.text:
                        new_upload_status_text = mid_status.text
                        print(f"NOTE: {new_upload_status_text}")
                        time.sleep(1)
                    continue

                # Give the upload status time to update
                time.sleep(3)

                if "Upload complete" in status.text:
                    print(f"NOTE: {status.text}")
                    success = True
                else:
                    print(f"ERROR: {status.text}")

        # Images cannot be uploaded because they already exists
        elif "Cannot upload any of these image files" in status.text:
            print(f"NOTE: {status.text}, they already exist in source.")
            success = True

        # Unexpected status
        else:
            print(f"Warning: {status.text}")

    except Exception as e:
        print(f"ERROR: Issue with uploading images. \n{e}")

    time.sleep(3)

    return driver, success


def upload_labelset(driver, source_id, labelset):
    """
    Upload labelsets to CoralNet.
    """

    print("\nNOTE: Navigating to labelset upload page")

    # Create a variable to track the success of the upload
    success = False

    # Go to the upload page
    driver.get(CORALNET_URL + f"/source/{source_id}/labelset/import/")

    # First check that this is existing source the user has access to
    try:
        # Check the permissions
        driver, status = check_permissions(driver)

        # Check the status
        if "Page could not be found" in status.text:
            raise Exception(f"ERROR: {status.text.split('.')[0]}")

    except Exception as e:
        print(f"ERROR: {e} or you do not have permission to access it")
        return driver, success

    # Check if files can be uploaded, get the status for the page
    path = "status_display"
    status = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, path)))

    # Get the status details for the page
    path = "status_detail"
    status_details = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, path)))

    try:
        # Locate the file input field
        path = "//input[@type='file'][@name='csv_file']"
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, path)))

        # Check if the file input field is enabled
        if file_input.is_enabled():
            # Submit the file
            print(f"NOTE: File input field is enabled")
            print(f"NOTE: Uploading {os.path.basename(labelset)}")
            file_input.send_keys(labelset)

            # Give the upload status time to update
            time.sleep(5)

            # Check the status
            if "Save labelset" in status.text:

                # Wait for the upload button to appear
                path = "//button[@id='id_upload_submit']"
                upload_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, path)))

                # Click the upload button
                upload_button.click()

                # Give the upload status time to update
                time.sleep(5)

                # Check the status,
                if "Labelset saved" in status.text:
                    print(f"NOTE: {status.text}")
                    success = True

            elif "Error" in status.text:
                # The file is not formatted correctly
                raise Exception(f"ERROR: {status.text}\n"
                                f"ERROR: {status_details.text}")
            else:
                # File input field is enabled, but something is wrong
                raise Exception(f"ERROR: Could not upload {labelset}")
        else:
            # File input field is not enabled, something is wrong
            raise Exception("ERROR: File input field is not enabled; exiting.")

    except Exception as e:
        print(f"ERROR: Could not submit file for upload\n{e}")

    time.sleep(3)

    return driver, success


def upload_annotations(driver, source_id, annotations):
    """
    Upload annotations to CoralNet.
    """

    print("\nNOTE: Navigating to annotation upload page")

    # Create a variable to track the success of the upload
    success = False

    # If there are already annotations, some will be overwritten
    alert = False

    # Go to the upload page
    driver.get(CORALNET_URL + f"/source/{source_id}/upload/annotations_csv/")

    # First check that this is existing source the user has access to
    try:
        # Check the permissions
        driver, status = check_permissions(driver)

        # Check the status, user doesn't have permission
        if "Page could not be found" in status.text:
            raise Exception(f"ERROR: {status.text.split('.')[0]} or you do not"
                            f" have permission to access it")

        # Check the status, source doesn't have a labelset yet
        if "create a labelset before uploading annotations" in status.text:
            raise Exception(f"ERROR: {status.text.split('.')[0]}")

    except Exception as e:
        print(f"{e}")
        return driver, success

    # Check if files can be uploaded, get the status for the page
    path = "status_display"
    status = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, path)))

    # Get the status details for the page
    path = "status_detail"
    status_details = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, path)))

    try:
        # Locate the file input field
        path = "//input[@type='file'][@name='csv_file']"
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, path)))

        # Check if the file input field is enabled
        if file_input.is_enabled():

            # Submit the file
            print(f"NOTE: File input field is enabled")
            print(f"NOTE: Uploading {os.path.basename(annotations)}")
            file_input.send_keys(annotations)

            print(f"NOTE: {status.text}")

            # Give the upload status time to update
            time.sleep(3)

            # Wait while the annotations are processing
            while "Processing" in status.text:
                continue

            # If there was an error, raise an exception
            if "Error" in status.text:
                # The file is not formatted correctly
                raise Exception(f"ERROR: {status.text}\n"
                                f"ERROR: {status_details.text}")

            # Data was sent successfully
            elif "Data OK" in status.text:

                # Print the status details
                print(f"\n{status_details.text}\n")

                if "deleted" in status_details.text:
                    alert = True

                # Wait for the upload button to appear
                path = "//button[@id='id_upload_submit']"
                upload_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, path)))

                # Click the upload button
                upload_button.click()

                # There are already annotations for these files, overwrite them
                if alert:
                    # Wait for the alert dialog to appear
                    alert = WebDriverWait(driver, 10).until(
                        EC.alert_is_present())

                    # Switch to the alert dialog
                    alert = driver.switch_to.alert

                    # Accept the alert (OK button)
                    alert.accept()

                    # Switch back to the main content
                    driver.switch_to.default_content()

                # Give the upload status time to update
                time.sleep(3)

                # Get the status for the page
                print(f"NOTE: {status.text}")

                # Wait while the annotations are saving
                while "Saving" in status.text:
                    continue

                # Check if the annotations were saved successfully
                if "saved" in status.text:
                    print(f"NOTE: {status.text}")
                    success = True
                else:
                    # The annotations were not saved successfully
                    raise Exception(f"ERROR: {status.text}"
                                    f"{status_details.text}")

        else:
            # File input field is not enabled, something is wrong
            raise Exception("ERROR: File input field is not enabled; exiting.")

    except Exception as e:
        print(f"ERROR: Could not upload annotations.\n{e}")

    time.sleep(3)

    return driver, success

# -----------------------------------------------------------------------------
# TODO include checks for the source, data formats, argparse, notebook
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='CoralNet arguments')

    parser.add_argument('--username', type=str,
                        default=os.getenv('CORALNET_USERNAME'),
                        help='Username for CoralNet account')

    parser.add_argument('--password', type=str,
                        default=os.getenv('CORALNET_PASSWORD'),
                        help='Password for CoralNet account')

    parser.add_argument('--source_id', type=int,
                        help='Source ID to upload to.')

    parser.add_argument('--images', type=str, default="",
                        help='The A directory where all images are located')

    parser.add_argument('--annotations', type=str, default="",
                        help='The path to the annotations file')

    parser.add_argument('--labelset', type=str, default="",
                        help='The path to the labelset file')

    parser.add_argument('--headless', type=str, default='True',
                        choices=['True', 'False'],
                        help='Run browser in headless mode')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Get the browser
    # -------------------------------------------------------------------------
    options = Options()

    if args.headless.lower() == 'true':
        # Set headless mode
        options.add_argument("--headless")

    # Pass the options object while creating the driver
    driver = check_for_browsers(options=options)

    # Flags to determine what to upload
    UPLOAD_IMAGES = False
    UPLOAD_LABELSET = False
    UPLOAD_ANNOTATIONS = False

    # Data to be uploaded
    IMAGES = os.path.abspath(args.images)
    IMAGES = glob.glob(IMAGES + "\\*.*")
    IMAGES = [i for i in IMAGES if i.split('.')[-1] in IMG_FORMATS]

    # Check if there are images to upload
    if len(IMAGES) > 0:
        print(f"NOTE: Found {len(IMAGES)} images to upload.")
        UPLOAD_IMAGES = True

    # Assign the labelset
    LABELSET = os.path.abspath(args.labelset)

    # Check if there is a labelset to upload
    if os.path.exists(LABELSET) and "csv" in LABELSET.split(".")[-1]:
        print(f"NOTE: Found labelset to upload.")
        UPLOAD_LABELSET = True

    # Assign the annotations
    ANNOTATIONS = os.path.abspath(args.annotations)

    # Check if there are annotations to upload
    if os.path.exists(ANNOTATIONS) and "csv" in ANNOTATIONS.split(".")[-1]:
        print(f"NOTE: Found annotations to upload.")
        UPLOAD_ANNOTATIONS = True

    # If there are no images, labelset, or annotations to upload, exit
    if not UPLOAD_IMAGES and not UPLOAD_LABELSET and not UPLOAD_ANNOTATIONS:
        print(f"ERROR: No data to upload. Please check the following files:\n"
              f"Images: {args.images}\n"
              f"Labelset: {args.labelset}\n"
              f"Annotations: {args.annotations}")
        sys.exit(1)

    # Credentials
    username = args.username
    password = args.password

    try:
        # Authenticate, make sure the credentials are correct
        authenticate(username, password)
        CORALNET_TOKEN, HEADERS = get_token(username, password)

    except Exception as e:
        print(f"ERROR: Could not authenticate for user {username}.\n{e}")

    # ID of the source to upload data to
    source_id = args.source_id

    # -------------------------------------------------------------------------
    # Upload the data
    # -------------------------------------------------------------------------

    # Log in to CoralNet
    driver, _ = login(username, password, driver)

    # Upload images
    if UPLOAD_IMAGES:
        driver, _ = upload_images(driver, source_id, IMAGES)

    # Upload labelset
    if UPLOAD_LABELSET:
        driver, _ = upload_labelset(driver, source_id, LABELSET)

    # Upload annotations
    if UPLOAD_ANNOTATIONS:
        driver, _ = upload_annotations(driver, source_id, ANNOTATIONS)

    # Close the browser
    print("NOTE: Closing connection...")
    driver.quit()

    print("Done.")


if __name__ == "__main__":
    main()
