import os
import glob
import sys
import time

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from CoralNet import *


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def login(driver):
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
    username_input.send_keys(USERNAME)
    password_input.send_keys(PASSWORD)

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
        raise ValueError(f"ERROR: Could not login with {USERNAME}\n{e}")

    return driver, success


def upload_images(driver, images):
    """
    Upload images to CoralNet.
    """

    print("\nNavigating to image upload page...")

    # Variable for success
    success = False

    # Go to the upload page
    driver.get(CORALNET_URL + f"/source/{SOURCE_ID}/upload/images/")

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
        status = WebDriverWait(driver, len(IMAGES)).until(
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


def upload_labelset(driver, labelset):
    """
    Upload labelsets to CoralNet.
    """

    print("\nNOTE: Navigating to labelset upload page")

    # Create a variable to track the success of the upload
    success = False

    # Go to the upload page
    driver.get(CORALNET_URL + f"/source/{SOURCE_ID}/labelset/import/")

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
                raise Exception(f"ERROR: {status.text}: {status_details.text}")
            else:
                # File input field is enabled, but something is wrong
                raise Exception(f"ERROR: Could not upload {LABELSET}")
        else:
            # File input field is not enabled, something is wrong
            raise Exception("ERROR: File input field is not enabled; exiting.")

    except Exception as e:
        print(f"ERROR: Could not submit file for upload\n{e}")

    time.sleep(3)

    return driver, success


def upload_annotations(driver, annotations):
    """
    Upload annotations to CoralNet.
    """

    print("\nNOTE: Navigating to annotation upload page")

    # Create a variable to track the success of the upload
    success = False

    # If there are already annotations, some will be overwritten
    alert = False

    # Go to the upload page
    driver.get(CORALNET_URL + f"/source/{SOURCE_ID}/upload/annotations_csv/")

    try:
        # First check that there is already a labelset uploaded
        path = "content-container"
        status = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Check the status
        if "create a labelset before uploading annotations" in status.text:
            raise Exception(f"ERROR: {status.text.split('.')[0]}")

    except Exception as e:
        print(f"ERROR: Could not upload annotations.\n{e}")
        # return success

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
                raise Exception(f"ERROR: {status.text}: {status_details.text}")

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

    # argparse.ArgumentParser()

    # Username
    USERNAME = os.getenv("CORALNET_USERNAME")

    # Password
    PASSWORD = os.getenv("CORALNET_PASSWORD")

    try:
        # Authenticate
        authenticate(USERNAME, PASSWORD)
        CORALNET_TOKEN, HEADERS = get_token(USERNAME, PASSWORD)
    except Exception as e:
        print(e)

    # ID of the source to upload data to
    SOURCE_ID = 4054

    # Where the data root is on the local machine
    # The path needs to be absolute, not relative.
    DATA_ROOT = "../CoralNet_Data/3420/"
    DATA_ROOT = os.path.abspath(DATA_ROOT) + "/"

    # Check that the data root exists
    if not os.path.exists(DATA_ROOT):
        raise ValueError(f"ERROR: {DATA_ROOT} does not exist")

    # Collecting data to be uploaded
    IMAGES = glob.glob(DATA_ROOT + "images/*.*")[0:25]
    LABELSET = DATA_ROOT + "labelset.csv"
    ANNOTATIONS = DATA_ROOT + "annotations.csv"

    # Flags to determine what to upload
    UPLOAD_IMAGES = False if len(IMAGES) == 0 else True
    UPLOAD_LABELSET = False if not os.path.exists(LABELSET) else True
    UPLOAD_ANNOTATIONS = False if not os.path.exists(ANNOTATIONS) else True

    # Check data formats locally before trying to upload, check source page
    #
    #
    #

    # Set up the browser driver (in this case, Chrome)
    driver = webdriver.Chrome()

    # Log in to CoralNet
    driver, _ = login(driver)

    # Upload images
    if UPLOAD_IMAGES:
        driver, _ = upload_images(driver, IMAGES)

    # Upload labelset
    if UPLOAD_LABELSET:
        driver, _ = upload_labelset(driver, LABELSET)

    # Upload annotations
    if UPLOAD_ANNOTATIONS:
        driver, _ = upload_annotations(driver, ANNOTATIONS)

    # Close the browser
    if input("\nExit? (y/n): ").lower() == "y":
        print("NOTE: Closing connection...")
        driver.quit()

    print("Done.")

if __name__ == "__main__":
    main()