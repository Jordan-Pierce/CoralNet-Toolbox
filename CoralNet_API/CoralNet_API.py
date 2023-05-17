import re
import time
import glob
import random
import argparse
import datetime
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

from CoralNet import *
from CoralNet_Download import *

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def in_N_seconds(wait):
    """
    Calculate the time in N seconds from the current time.

    Args:
    - wait: an integer representing the number of seconds to wait

    Returns:
    - A string representing the time in `wait` seconds from the current time,
    in the format "HH:MM:SS"
    """
    # Get the current time, and add the wait time
    now = datetime.datetime.now()
    then = now + datetime.timedelta(seconds=wait)
    return then.strftime("%H:%M:%S")


def is_expired(url):
    """
    Calculates the time remaining before a URL expires, based on its "Expires"
    timestamp.

    Args:
    url (str): The URL to check.
    """
    # Assume the URL is expired
    expired = True
    # Set the time remaining to 0
    time_remaining = 0

    try:
        # Extract expiration timestamp from URL
        match = re.search(r"Expires=(\d+)", url)

        # If the timestamp was found, extract it
        if match:
            # Convert the timestamp to an integer
            expiration = int(match.group(1))

            # Calculate time remaining before expiration
            time_remaining = expiration - int(time.time())
        else:
            raise ValueError(f"ERROR: Could not find expiration timestamp "
                             f"in \n{url}")

    except Exception as e:
        print(f"{e}")

    # Check the amount of time remaining
    if time_remaining >= 200:
        expired = False

    return expired


def sample_points_for_url(url, num_samples=200, method='stratified'):
    """
    Generates a set of sample coordinates within a given image size.

    Parameters:
    ----------
    width : int
        The width of the image.
    height : int
        The height of the image.
    num_samples : int, optional
        The number of samples to generate. Default is 200.
    method : str, optional
        The method to use for generating samples. Valid values are:
        - 'uniform': generates samples using uniform sampling
        - 'random': generates samples using random sampling
        - 'stratified': generates samples using stratified sampling (default)

    Returns:
    -------
    tuple
        A tuple containing three elements:
        - A numpy array of x-coordinates of the generated samples.
        - A numpy array of y-coordinates of the generated samples.
        - A list of dictionaries containing row and column coordinates of the
        generated samples.
    """

    x = None
    y = None
    samples = None

    # Check if the URL is expired
    if is_expired(url):
        print(f"ERROR: URL is expiring soon; skipping.\n{url}")

    else:
        # Request the image from AWS
        response = requests.get(url)

        # Read it to get the size
        img = Image.open(BytesIO(response.content))
        width, height = img.size

        x_coordinates = []
        y_coordinates = []
        samples = []

        # Generate samples
        if method == 'uniform':
            x_coords = np.linspace(0, width - 1, int(np.sqrt(num_samples)))
            y_coords = np.linspace(0, height - 1, int(np.sqrt(num_samples)))
            for x in x_coords:
                for y in y_coords:
                    x_coordinates.append(int(x))
                    y_coordinates.append(int(y))
                    samples.append({'row': int(y), 'column': int(x)})
        # Generate samples
        elif method == 'random':
            for i in range(num_samples):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                x_coordinates.append(x)
                y_coordinates.append(y)
                samples.append({'row': y, 'column': x})
        # Generate samples
        elif method == 'stratified':
            n = int(np.sqrt(num_samples))
            x_range = np.linspace(0, width - 1, n + 1)
            y_range = np.linspace(0, height - 1, n + 1)
            for i in range(n):
                for j in range(n):
                    x = np.random.uniform(x_range[i], x_range[i + 1])
                    y = np.random.uniform(y_range[j], y_range[j + 1])
                    x_coordinates.append(int(x))
                    y_coordinates.append(int(y))
                    samples.append({'row': int(y), 'column': int(x)})
        # Store in numpy arrays
        x = np.array(x_coordinates).astype(int)
        y = np.array(y_coordinates).astype(int)

    return x, y, samples


def check_job_status(response, coralnet_token):
    """
    Sends a request to retrieve the completed annotations and returns the
    status update.

    Parameters:
    ----------
    response : requests.Response
        A Response object returned from a previous request to CoralNet API.

    Returns:
    -------
    dict
        A dictionary containing status information, which includes these keys:
        - 'status': a string indicating the current status
        - 'message': a string providing additional details about the job status
    """

    # Create the payload
    url = f"https://coralnet.ucsd.edu{response.headers['Location']}"
    headers = {"Authorization": f"Token {coralnet_token}"}
    # Sends a request to retrieve the completed annotations
    status = requests.get(url=url, headers=headers)
    # Convert the response to JSON
    current_status = json.loads(status.content)
    wait = 1

    if status.ok:
        # Still in progress
        if 'status' in current_status['data'][0]['attributes'].keys():
            # Extract the status information
            s = current_status['data'][0]['attributes']['successes']
            f = current_status['data'][0]['attributes']['failures']
            t = current_status['data'][0]['attributes']['total']
            status_str = current_status['data'][0]['attributes']['status']
            ids = current_status['data'][0]['id'].split(",")
            ids = ''.join(str(_) for _ in ids)
            # Get the current time
            now = time.strftime("%H:%M:%S")
            # Create the message
            message = f"Status: {status_str}\tID: {ids}\tTime: {now}"
            message += f"\tTotal: {t}\tSuccess: {s}\tFailures: {f}"

        else:
            # It's done with all images
            message = "Completed Job"
    else:
        # CoralNet is getting too many requests, sleep for a second.
        message = f"CoralNet: {current_status['errors'][0]['detail']}"
        try:
            # Try to wait the amount of time requested by CoralNet
            match = re.search(r'\d+', message)
            wait = int(match.group())
        except:
            wait = 30

    return current_status, message, wait


def print_job_status(queue, active, completed, expired):
    """
    Print the current status of jobs and images being processed.

    Args:
    - queued_jobs (list): A list of jobs that are currently queued.
    - active_jobs (list): A list of jobs that are currently active.
    - completed_jobs (list): A list of jobs that have been completed.
    - expired_images (list): A list of images that need updated URLs.
    """
    print(f"JOBS: Queued: {len(queue)} \t"
          f"Active: {len(active)} \t"
          f"Completed: {len(completed)} \t"
          f"Expired Images: {len(expired)}")


def convert_to_csv(status, image_names, output_dir):
    """
    Converts response data into a Pandas DataFrame and concatenates each row
    into a single DataFrame.

    Parameters:
    ----------
    response : dict
        A dictionary object containing response data from a server.
    image_file : str
        The name of the image file corresponding to the response data.

    Returns:
    -------
    model_predictions : pandas.DataFrame
        A Pandas DataFrame containing prediction data
    """

    print(f"NOTE: Saving annotations for {len(image_names)} images")

    # A list to store all the model predictions (dataframes)
    model_predictions = []
    expired = []

    for (data, image_name) in list(zip(status['data'], image_names)):

        # Check if the image has expired after uploading to CoralNet
        if 'errors' in data['attributes']:
            expired.append(image_name)
            continue

        # Create a DataFrame to store the model predictions
        model_prediction = pd.DataFrame()
        # Loop through each point in the response
        for point in data['attributes']['points']:
            # Create a dictionary to store the data for each point
            p = dict()
            p['image_name'] = image_name
            p['column'] = point['column']
            p['row'] = point['row']

            # Loop through each classification for each point
            for index, classification in enumerate(point['classifications']):
                p['score_' + str(index + 1)] = classification['score']
                p['label_id_' + str(index + 1)] = classification['label_id']
                p['label_code_' + str(index + 1)] = classification['label_code']
                p['label_name_' + str(index + 1)] = classification['label_name']

            # Concatenate the data for each point into a single DataFrame
            p = pd.DataFrame.from_dict([p])
            model_prediction = pd.concat([model_prediction, p])

        # Save the model predictions to a CSV file
        basename = os.path.basename(image_name).split(".")[0]
        output_file = output_dir + basename + ".csv"
        model_prediction.reset_index(drop=True, inplace=True)
        model_prediction.to_csv(output_file, index=True)

        if os.path.exists(output_file):
            model_predictions.append(model_prediction)

    print(f"NOTE: Saved {len(model_predictions)} predictions to CSV files\n"
          f"NOTE: {len(expired)} images had expired and will be re-uploaded")

    return model_predictions, expired


# -----------------------------------------------------------------------------
# Main Function
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

    parser.add_argument('--csv_path', type=str,
                        help='A path to a csv file, or folder containing '
                             'multiple csv files. Each csv file should '
                             'contain following: image_name, row, column')

    parser.add_argument('--source_id', type=int,
                        help='The ID of the source being used.')

    parser.add_argument('--output_dir', type=str, default="../CoralNet_Data/",
                        help='A root directory where all predictions will be '
                             'saved to.')

    args = parser.parse_args()

    try:
        # Check to see if the csv file exists
        assert os.path.exists(args.csv_path)
        # Determine if it's a single file or a folder
        if os.path.isfile(args.csv_path):
            # If file, just read it in
            DATA = pd.read_csv(args.csv_path)
        elif os.path.isdir(args.csv_path):
            # If folder, read in all csv files, concatenate them together
            csv_files = glob.glob(args.csv_path + "/*.csv")
            DATA = pd.DataFrame()
            for csv_file in csv_files:
                DATA = pd.concat([DATA, pd.read_csv(csv_file)])
        else:
            raise Exception(f"ERROR: {args.csv_path} is invalid.")

        # Check to see if the csv file has the expected columns
        assert 'image_name' in DATA.columns

    except Exception as e:
        print(f"ERROR: File(s) provided do not match expected format!\n"
              f"{args.csv_path}\n{e}")
        sys.exit(1)

    # Username, Password
    USERNAME = args.username
    PASSWORD = args.password

    try:
        # Authenticate
        authenticate(USERNAME, PASSWORD)
        CORALNET_TOKEN, HEADERS = get_token(USERNAME, PASSWORD)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Source information
    SOURCE_ID = str(args.source_id)

    # Variables for the model
    metadata = get_model_meta(SOURCE_ID, USERNAME, PASSWORD)
    # Check if the model exists
    if metadata is None:
        raise Exception(f"ERROR: No model found for the source {SOURCE_ID}.")

    # Get the model ID and URL
    MODEL_ID = metadata['Model_ID'][0]
    MODEL_URL = CORALNET_URL + f"/api/classifier/{MODEL_ID}/deploy/"

    # All images associated with the source
    SOURCE_IMAGES = get_images(SOURCE_ID, USERNAME, PASSWORD)

    # Check if there are any images
    if SOURCE_IMAGES is None:
        raise Exception(f"ERROR: No images found in the source {SOURCE_ID}.")

    # Set the data root directory
    DATA_ROOT = args.output_dir + "/"

    # Where the output predictions will be stored
    SOURCE_DIR = DATA_ROOT + SOURCE_ID + "/"
    SOURCE_POINTS = SOURCE_DIR + "points/"
    SOURCE_PREDICTIONS = SOURCE_DIR + "predictions/"

    # Create a folder to contain predictions and points
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(SOURCE_POINTS, exist_ok=True)
    os.makedirs(SOURCE_PREDICTIONS, exist_ok=True)

    # Get the images desired for predictions; make sure it's not file path.
    images = DATA['image_name'].unique().tolist()
    images = [os.path.basename(image) for image in images]

    try:
        # We will get the information needed from the source images dataframe
        IMAGES = SOURCE_IMAGES[SOURCE_IMAGES['image_name'].isin(images)]
        print(f"NOTE: Found {len(IMAGES)} images in the source {SOURCE_ID}.")
    except Exception as e:
        print(f"ERROR: Image names in {args.csv_path} do not match any of "
              f"those in the source {SOURCE_ID}. Make sure they have already "
              f"been uploaded to the source before using the API script.\n{e}")
        sys.exit(1)

    # Check if we have row and column information available, else sample points
    if 'row' not in DATA.columns or 'column' not in DATA.columns:
        if input(f"NOTE: No row or column information was found in "
                 f"{args.csv_path}\n. Do you want randomly sample 200 "
                 f"points? (y/n): ").lower() == 'n':
            # Exit the script
            print("NOTE: Exiting script.")
            sys.exit(0)

        # Create points for each of the desired images
        for image in images:
            # We use the IMAGES dataframe to get the URL of the image
            image_df = IMAGES[IMAGES['image_name'] == image]
            image_url = image_df['image_url'].values[0]
            # Then we sample points from the image
            x, y, samples = sample_points_for_url(image_url, 200, 'stratified')
            # Save the points to a csv file in the SOURCE_POINTS folder
            df = pd.DataFrame(samples); df['image_name'] = image
            df.to_csv(SOURCE_POINTS + image + ".csv")
            if os.path.exists(SOURCE_POINTS + image + ".csv"):
                print(f"NOTE: Points for {image} saved successfully")
            else:
                print(f"ERROR: Could not save points for {image}")

        # Get all the points for all the images that were just created
        POINT_PATHS = glob.glob(SOURCE_POINTS + "*.csv")

        # This dataframe will contain all the points for all the images
        # The columns are `image_name`, `Row`, and `Column`.
        POINTS = pd.DataFrame()
        # We then concatenate all the points into a single dataframe
        for path in POINT_PATHS:
            points = pd.read_csv(path)
            points['image_name'] = os.path.basename(path)
            POINTS = pd.concat([POINTS, points])

    else:
        # We will use the points provided in the CSV file
        POINTS = DATA[['image_name', 'row', 'column']]

    # ---------------------------------------------------------------------------------------------
    # We will now get the predictions for each of the images
    # ---------------------------------------------------------------------------------------------
    # Jobs that are currently queued
    queued_jobs = []
    queued_imgs = []
    # Jobs that are currently active
    active_jobs = []
    active_imgs = []
    # Jobs that are completed
    completed_jobs = []
    completed_imgs = []
    # A list that contains just the images that need updated urls
    expired_imgs = []
    # Flag to indicate if all images have been passed to model
    finished = False
    # The amount of time to wait before checking the status of a job
    patience = 75
    # The number of images to include in each job
    data_per_payload = 100

    # This will continue looping until all images have been processed
    while not finished:

        # A list for images and data that have been sampled this round
        payload_data = []
        payload_imgs = []

        for index, row in IMAGES.iterrows():
            # Loops through each image requested, gets points, adds to a queue

            # Get the current image name and url
            name = row['image_name']
            url = row['image_url']

            # If this image has already been completed, skip it.
            if name in completed_imgs:
                # print(f"Image {name} already completed; skipping")
                continue  # Skip to the next image within the current for loop

            # If this image is already in active, skip it.
            elif any(name in n for n in active_imgs):
                # print(f"Image {name} already active; skipping")
                continue  # Skip to the next image within the current for loop

            # if this image is already in queued, skip it.
            elif any(name in n for n in queued_imgs):
                # print(f"Image {name} already queued; skipping")
                continue  # Skip to the next image within the current for loop

            # The image url has not expired, so we can queue the image
            elif not is_expired(url):
                print(f"NOTE: Image {name} not in queue; sampling")
                points = POINTS[POINTS['image_name'].str.contains(name)]
                points = points[['row', 'column']].to_dict(orient="records")

                # Add the data to the list for payloads
                payload_imgs.append(name)
                payload_data.append(
                    {
                        "type": "image",
                        "attributes": {
                            "name": name,
                            "url": url,
                            "points": points
                        }
                    })

            else:
                # The image url expired, so we need to update it later.
                print(f"WARNING: {name} expired; adding to expired list")
                expired_imgs.append(name)
                continue  # Skip to the next image within the current for loop

        # Here we initialize the payload, which is a JSON object that
        # contains the image URLs and their points; payloads will contain
        # batches of data (N = data_per_payload).
        for _ in np.arange(0, len(payload_imgs), data_per_payload):
            # Get the image names and data for the payload
            image_names = payload_imgs[_: _ + data_per_payload]
            payload = {'data': payload_data[_: _ + data_per_payload]}
            # Use the payload to construct the job
            job = {
                "headers": HEADERS,
                "model_url": MODEL_URL,
                "image_names": image_names,
                "data": json.dumps(payload, indent=4),

            }
            # Add the job to the queue
            queued_jobs.append(job)
            queued_imgs.append(image_names)

        # Print the status of the jobs
        print_job_status(queued_jobs, active_jobs, completed_jobs, expired_imgs)

        # Start uploading the queued jobs to CoralNet if there are
        # less than 5 active jobs, and there are more in the queue.
        # If there are no queued jobs, this won't need to be entered.
        while len(active_jobs) < 5 and len(queued_jobs) > 0:

            # Loop through all the queued jobs
            for job, names in list(zip(queued_jobs, queued_imgs)):

                # Break when active gets to 5
                if len(active_jobs) >= 5:
                    print("NOTE: Five jobs already active; checking status")
                    break  # Breaks from both loops

                # Upload the image and the sampled points to CoralNet
                print(f"NOTE: Attempting to upload {len(names)} images")

                # Sends the requests to the `source` and in exchange, receives
                # a message telling if it was received correctly.
                response = requests.post(url=job["model_url"],
                                         data=job["data"],
                                         headers=job["headers"])
                if response.ok:
                    # If it was received
                    print(f"NOTE: Successfully uploaded: {len(names)} images")

                    # Add to active jobs
                    active_jobs.append(response)
                    active_imgs.append(names)

                    # Remove from queued jobs
                    queued_jobs.remove(job)
                    queued_imgs.remove(names)

                else:
                    # There was an error uploading to CoralNet; get the message
                    message = json.loads(response.text)['errors'][0]['detail']

                    # Print the message
                    print(f"CoralNet: {message}")

                    if "5 jobs active" in message:
                        print(f"NOTE: Will attempt again at {in_N_seconds(patience)}")
                        time.sleep(patience)

                    else:
                        # Assumed that the images have expired
                        print(f"ERROR: Failed to upload: {len(names)} images")

                        # Add to expired images
                        expired_imgs.extend(names)

                        # Remove from queue
                        queued_jobs.remove(job)
                        queued_imgs.remove(names)

            # If all images have expired, break from the loop
            if IMAGES['image_name'].isin(expired_imgs).all():
                print("NOTE: All images have expired")
                break

        # Check the status of the active jobs, break when another can be added
        while len(active_jobs) <= 5 and len(active_jobs) != 0:

            # Check the status of the active jobs
            print_job_status(queued_jobs, active_jobs, completed_jobs, expired_imgs)

            # Sleep before checking status again
            print(f"\nNOTE: Checking status again at {in_N_seconds(patience)}")
            time.sleep(patience)

            # Loop through the active jobs
            for i, (job, names) in enumerate(list(zip(active_jobs, active_imgs))):

                # Check the status of the current job
                current_status, message, wait = check_job_status(job, CORALNET_TOKEN)

                # Print the message
                print(f"NOTE: {message}")

                # Current job finished, output the results, remove from queue
                if message == "Completed Job":

                    # Convert to csv, and save locally, check expired
                    predictions, expired = convert_to_csv(current_status,
                                                          names,
                                                          SOURCE_PREDICTIONS)

                    # Deal with images after the job has been completed
                    for name in names:
                        # If the image was in expired, add to expired
                        if name in expired:
                            expired_imgs.append(name)
                        # Else, add to completed_imgs
                        else:
                            completed_imgs.append(name)

                    # Add to completed jobs list
                    print(f"NOTE: Adding {len(names)} images to completed")
                    completed_jobs.append(current_status)

                    # Remove from active jobs, images list
                    print(f"NOTE: Removing {len(names)} images from active")
                    active_imgs.remove(names)
                    active_jobs.remove(job)

                # Wait for the specified time before checking the status again
                time.sleep(wait)

            # After checking the current status, break if another can be added
            # Else wait and check the status of the active jobs again.
            if len(active_jobs) < 5 and len(queued_jobs) > 0:
                print(f"NOTE: Active jobs {len(active_jobs)}; adding another.")
                break

        # Check to see everything has been completed, breaking the loop
        if not queued_jobs and not active_jobs and not expired_imgs:
            print("NOTE: All images have been processed; exiting loop.")
            finished = True

        # If there are no queued jobs, and no active jobs, but there are
        # images in expired, get just the AWS URL for the expired images and
        # update dataframe.
        if not queued_jobs and not active_jobs and expired_imgs:
            print(f"NOTE: Updating {len(expired_imgs)} expired images' URL")
            # Get in expired images in dataframe
            IMAGES = IMAGES[IMAGES['image_name'].isin(expired_imgs)].copy()
            old_urls = IMAGES['image_page'].tolist()
            new_urls = get_image_urls(old_urls, USERNAME, PASSWORD)
            IMAGES['image_url'] = new_urls
            expired_imgs = []


if __name__ == "__main__":
    main()

