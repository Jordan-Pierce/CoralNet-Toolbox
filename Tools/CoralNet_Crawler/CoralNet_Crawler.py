import os
import io

import requests
import pandas as pd
import multiprocessing
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

# Constant for the CoralNet url
CORALNET_URL = "https://coralnet.ucsd.edu"

# URL of the login page
LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"

# The source id of the source you want to download all the images from
SOURCE_ID = 2733

# The source id of the source you want to download all the images from
SOURCE_URL = CORALNET_URL + "/source/" + str(SOURCE_ID)
IMAGE_URL = SOURCE_URL + "/browse/images/"
LABELSET_URL = SOURCE_URL + "/export/labelset/"

# The directory to store the output
SOURCE_DIR = "./" + str(SOURCE_ID) + "/"
IMAGE_DIR = SOURCE_DIR + "images/"

# Creating the directories
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Set the username and password for your CoralNet account locally, that way
# credentials never need to be entered in the script (wherever it is).

# Be sure to restart the terminal when first setting environmental variables
# for them to be saved!
USERNAME = os.getenv('CORALNET_USERNAME')
PASSWORD = os.getenv('CORALNET_PASSWORD')


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def url_to_soup(url):
    """
    Takes a URL and returns a BeautifulSoup object for the HTML at the URL.

    Args: url (str): The URL of the webpage to be converted to a
    BeautifulSoup object.

    Returns:
        soup (BeautifulSoup): The BeautifulSoup object for the HTML at the URL.
    """

    # Send an HTTP GET request to the URL and store the response
    response = requests.get(url)

    # Parse the HTML, store in soup
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    return soup


def crawl(page_url):
    """
    Crawl a coralnet image page and get the associated image path url and
    the url of the next image page.

    Args:
        page_url (str): The URL of the coralnet image page to crawl.

    Returns: tuple: A tuple containing the image page URL, image path URL,
    and the URL of the next image page.
    """

    image_name = None
    image_page_url = page_url
    image_path_url = None
    next_image_page_url = None

    # From the provided image page url, get the associated image path url
    soup = url_to_soup(page_url)

    # Find the div element with id="original_image_container" and
    # style="display:none;"
    orginal_image_container = soup.find('div',
                                        id='original_image_container',
                                        style='display:none;')

    # Find the img element within the div and get the value of the src
    # attribute
    image_path_url = orginal_image_container.find('img').get('src')

    # Now, get the next page's url
    for a_tag in soup.find_all('a'):
        # check if the text of the <a> tag contains "Next"
        if "Next" in a_tag.text:
            # Get the value of the href attribute and prepend the CORALNET_URL
            next_image_page_url = CORALNET_URL + a_tag.get('href')

        # Else, it returns None, and we know we're at the end of the images

    # Finally, get the name of the image, because when downloaded it might
    # not match
    image_name = soup.find('title').text.split(" |")[0]

    return image_name, image_page_url, image_path_url, next_image_page_url


def download_labelset(labelset_url):
    """
    Downloads a .csv file holding the label set from the given URL.
    Args:
        labelset_url (str): The URL of the website with the download button.
    Returns:
        None
    """

    print("Downloading labelset...")

    # Make an HTTP GET request to download the .csv file
    response = requests.get(labelset_url,
                            params={"submit": "Export label entries to CSV"})

    # Check the response status code
    if response.status_code == 200:
        # Save the .csv file to a local directory
        with open(SOURCE_DIR + "label_set.csv", "wb") as f:
            f.write(response.content)

    if os.path.exists(SOURCE_DIR + "label_set.csv"):
        print("Label set saved to: ", SOURCE_DIR + "label_set.csv")
    else:
        print("Could not download label set.")


def download_annotations(image_url):
    """
    This function downloads the annotations from a CoralNet source using the
    provided image URL. It logs into CoralNet using the provided username
    and password, and exports the annotations for the images in the source
    as a CSV file, which is saved in the local source directory.

    :param image_url: A string containing the URL of an image in the source
    """

    print("Downloading annotations...")

    # Send a GET request to the login page to retrieve the login form
    response = requests.get(LOGIN_URL)

    # Pass along the cookies
    cookies = response.cookies

    # Parse the HTML of the response using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the CSRF token from the HTML of the login page
    csrf_token = soup.find("input", attrs={"name": "csrfmiddlewaretoken"})

    # Create a dictionary with the login form fields and their values
    # (replace "username" and "password" with your actual username and
    # password)
    data = {
        "username": USERNAME,
        "password": PASSWORD,
        "csrfmiddlewaretoken": csrf_token["value"],
    }

    # Include the "Referer" header in the request
    headers = {
        "Referer": LOGIN_URL,
    }

    # Use requests.Session to create a session that will maintain your login
    # state
    with requests.Session() as session:

        # Use session.post() to submit the login form, including the
        # "Referer" header
        response = session.post(LOGIN_URL,
                                data=data,
                                headers=headers,
                                cookies=cookies)

        # Use session.get() to make a GET request to the source URL
        response = session.get(image_url)

        # Pass along the cookies
        cookies = response.cookies

        # Parse the HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the form in the HTML
        form = soup.find("form", {"id": "export-annotations-form"})

        # Extract the form fields (input elements)
        inputs = form.find_all("input")

        # Create a dictionary with the form fields and their values
        data = {'optional_columns': []}
        for i, input in enumerate(inputs):
            if i == 0:
                data[input["name"]] = input["value"]
            else:
                data['optional_columns'].append(input['value'])

        # Use session.post() to submit the form
        response = session.post(CORALNET_URL + form["action"],
                                data=data,
                                headers=headers,
                                cookies=cookies)

        # Check the response status code
        if response.status_code == 200:
            # Convert the text in response to a dataframe
            df = pd.read_csv(io.StringIO(response.text), sep=",")
            # Save the dataframe locally
            df.to_csv(SOURCE_DIR + "annotations.csv")

        else:
            print("Could not connect with CoralNet; please ensure that you "
                  "entered your username, password, and source ID correctly.")

    if os.path.exists(SOURCE_DIR + "label_set.csv"):
        print("Annotations saved to: ", SOURCE_DIR + "annotations.csv")
    else:
        print("Could not download annotations.")


def download_image(row):
    """
    Downloads a single image from the given URL, using the provided file name
    to save the image to a local directory.

    Args:
        row (list): A list containing the file name and URL of the image to
            download. The list should have at least two elements: the file
            name and the URL.

    Returns:
        None
    """

    # the output path of the image being downloaded
    image_path = IMAGE_DIR + row[0]

    # Make an HTTP GET request to download the image
    response = requests.get(row[2])

    # Check the response status code
    if response.status_code == 200:
        # Save the image to a file in the output directory
        with open(image_path, 'wb') as f:
            f.write(response.content)

        if os.path.exists(image_path):
            print("File downloaded: ", image_path)
        else:
            print("File could not be downloaded: ", image_path)


def download_images(image_url):
    """
    Downloads images from a list of URLs, using file names from a pandas
    dataframe to save the images to a local directory.

    Args: df (pandas.DataFrame): A dataframe containing the file names and
    URLs of the images to download. The dataframe should have at least two
    columns: 'file_name' and 'image_url'.

    Returns:
        None
    """

    print("Downloading images...")

    # A list containing the urls to all the image pages and a list containing
    # the urls to all the images hosted on amazon
    image_names = []
    image_page_urls = []
    image_path_urls = []

    # Convert the webpage to soup
    soup = url_to_soup(image_url)

    # Grab the first image page url
    images_divs = soup.find('span', class_='thumb_wrapper')
    image_href = images_divs.find_all('a')[0].get("href")
    next_page_url = CORALNET_URL + image_href

    # Crawl across all image page urls, grabbing the image path url as well
    # as the next page url continue doing this until the end of the source
    # project image pages, where there is no next.
    while next_page_url is not None:
        name, page_url, path_url, next_page_url = crawl(next_page_url)

        image_names.append(name)
        image_page_urls.append(page_url)
        image_path_urls.append(path_url)

        print(name, page_url, path_url)

    # Storing the results in dataframe
    df = pd.DataFrame(list(zip(image_names, image_page_urls, image_path_urls)),
                      columns=['image_name', 'image_page', 'image_url'])
    # Saving locally
    df.to_csv(SOURCE_DIR + str(SOURCE_ID) + "_images.csv")

    # Use the multiprocessing library's Pool.map() method to download images
    # in parallel
    with multiprocessing.Pool() as pool:
        # Apply the download_image function to each row in the dataframe
        pool.map(download_image, df.values.tolist())


if __name__ == "__main__":

    """This is the main function of the script. It calls the functions 
    download_labelset, download_annotations, and download_images to download 
    the label set, annotations, and images, respectively. """

    # Download the label set as a csv
    download_labelset(LABELSET_URL)

    # Check to see if user credentials have been set,
    # if not, annotations cannot be downloaded; skip.
    if not None in [USERNAME, PASSWORD]:
        # Download the annotations as a csv
        download_annotations(IMAGE_URL)

    # Download all the images
    download_images(IMAGE_URL)

    print("Done.")
