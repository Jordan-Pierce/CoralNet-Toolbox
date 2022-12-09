import os

import requests
import pandas as pd
import multiprocessing
from bs4 import BeautifulSoup


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

# Constant for the coralnet url
CORALNET_URL = "https://coralnet.ucsd.edu"

# The source id of the source you want to download all the images from
SOURCE_ID = 2687

# The directory to store the output
SOURCE_DIR = "./" + str(SOURCE_ID) + "/"
IMAGE_DIR = SOURCE_DIR + "images/"

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


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


def crawl(url):
    """
    Crawl a coralnet image page and get the associated image path url and
    the url of the next image page.

    Args:
        url (str): The URL of the coralnet image page to crawl.

    Returns: tuple: A tuple containing the image page URL, image path URL,
    and the URL of the next image page.
    """

    image_name = None
    image_page_url = url
    image_path_url = None
    next_image_page_url = None

    # From the provided image page url, get the associated image path url
    soup = url_to_soup(url)

    # Find the div element with id="original_image_container" and
    # style="display:none;"
    orginal_image_container = soup.find('div', id='original_image_container',
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

        # Else, it return None and we know we're at the end

    # Finally, get the name of the image, because when downloaded it might
    # not match
    image_name = soup.find('title').text.split(" |")[0]

    return image_name, image_page_url, image_path_url, next_image_page_url


def download_labelset(url):
    """
    Downloads a .csv file holding the labelset from the given URL.

    Args:
        url (str): The URL of the website with the download button.

    Returns:
        None
    """

    # Make an HTTP GET request to download the .csv file
    response = requests.get(url, params={"submit": "Export label entries to "
                                                   "CSV"})

    # Check the response status code
    if response.status_code == 200:
        # Save the .csv file to a local directory
        with open(SOURCE_DIR + "label_set.csv", "wb") as f:
            f.write(response.content)


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


def download_images(df):
    """
    Downloads images from a list of URLs, using file names from a pandas
    dataframe to save the images to a local directory.

    Args: df (pandas.DataFrame): A dataframe containing the file names and
    URLs of the images to download. The dataframe should have at least two
    columns: 'file_name' and 'image_url'.

    Returns:
        None
    """

    # Use the multiprocessing library's Pool.map() method to download images
    # in parallel
    with multiprocessing.Pool() as pool:
        # Apply the download_image function to each row in the dataframe
        pool.map(download_image, df.values.tolist())


if __name__ == "__main__":

    """This is the main function of the script. It initializes the necessary 
    constants and lists, converts the first image page URL to soup, and then 
    crawls through all the image pages to get the image page URLs and image 
    path URLs. """

    # A list containing the urls to all the image pages and a list containing
    # the the urls to all the images hosted on amazon
    image_names = []
    image_page_urls = []
    image_path_urls = []

    # The source id of the source you want to download all the images from
    source_url = CORALNET_URL + "/source/" + str(SOURCE_ID)
    image_url = source_url + "/browse/images/"
    labelset_url = source_url + "/export/labelset/"
    annotation_url = source_url + "/export/annotations/"

    # First download the labelset
    download_labelset(labelset_url)

    # Convert the webpage to soup
    soup = url_to_soup(image_url)

    # Grab the first image page url
    images_divs = soup.find('span', class_='thumb_wrapper')
    next_image_page_url = CORALNET_URL + images_divs.find_all('a')[0].get(
        "href")

    # Crawl across all image page urls, grabbing the image path url as well
    # as the next page url continue doing this until the end of the source
    # project image pages, where there is no next.
    while next_image_page_url is not None:
        image_name, image_page_url, image_path_url, next_image_page_url = crawl(
            next_image_page_url)

        image_names.append(image_name)
        image_page_urls.append(image_page_url)
        image_path_urls.append(image_path_url)

        print(image_name, image_page_url, image_path_url)

    # Storing the results in dataframe, saving locally
    df = pd.DataFrame(list(zip(image_names, image_page_urls, image_path_urls)),
                      columns=['image_name', 'image_page', 'image_url'])

    df.to_csv(SOURCE_DIR + str(SOURCE_ID) + "_images.csv")

    # Printing out the results
    print("\n", "Data scraped from CoralNet: ")
    print(df, "\n")

    # Downloading the images in parallel using multiprocessing
    print("Downloading images to: ", IMAGE_DIR)
    download_images(df)

    print("Done.")

