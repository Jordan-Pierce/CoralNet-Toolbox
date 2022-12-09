import requests
import pandas as pd
from bs4 import BeautifulSoup


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

# Constant for the coralnet url
CORALNET_URL = "https://coralnet.ucsd.edu"

# The source id of the source you want to download all the images from
SOURCE_ID = 665


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


if __name__ == "__main__":

    """This is the main function of the script. It initializes the necessary 
    constants and lists, converts the first image page URL to soup, and then 
    crawls through all the image pages to get the image page URLs and image 
    path URLs. """

    # Constant for the coralnet url
    CORALNET_URL = "https://coralnet.ucsd.edu"

    # A list containing the urls to all the image pages and a list containing
    # the the urls to all the images hosted on amazon
    image_names = []
    image_page_urls = []
    image_path_urls = []

    # The source id of the source you want to download all the images from
    source_url = CORALNET_URL + "/source/" + str(SOURCE_ID) + "/browse/images/"

    # Convert the webpage to soup
    soup = url_to_soup(source_url)

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

    print("Number of image pages: ", len(image_page_urls))
    print("Number of image paths: ", len(image_path_urls))

    df = pd.DataFrame(list(zip(image_names, image_page_urls, image_path_urls)),
                      columns=['image_name', 'image_page', 'image_url'])
    print(df)
    print("Done.")

