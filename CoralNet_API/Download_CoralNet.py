import os
import io
import sys
import os.path
import argparse
import requests
import pandas as pd
import multiprocessing
from bs4 import BeautifulSoup

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


def download_labelset(labelset_url, source_dir):
    """
    Downloads a .csv file holding the label set from the given URL.
    Args:
        labelset_url (str): The URL of the website with the download button.
    Returns:
        None
    """

    print("Downloading labelset...")

    # Make an HTTP GET request to download the .csv file
    response = requests.get(labelset_url)

    # Convert the HTML response to soup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the table we're looking for
    table = soup.find("table", class_="detail_table")

    labelset = []

    # Loop through each row in the table, ignoring the header
    for tr in table.find_all("tr")[1:]:
        label_id = tr.find("a").get("href").split("/")[-2]
        url = CORALNET_URL + tr.find("a").get("href")
        name = tr.find("a").text.strip()
        short_code = tr.find_all("td")[1].text.strip()
        functional_group = tr.find_all("td")[2].text.strip()

        labelset.append([label_id, url, name, short_code, functional_group])

    # Create dataframe to hold all labelset information
    df = pd.DataFrame(labelset, columns=['Label_ID', 'Label_URL', 'Name',
                                         'Short_Code', 'Functional_Group'])
    # Save locally
    df.to_csv(source_dir + "labelset.csv")

    if os.path.exists(source_dir + "labelset.csv"):
        print("Label set saved to: ", source_dir + "labelset.csv")
    else:
        print("Could not download labelset.")

    return df


def download_model_meta(source_url, source_dir):
    """This function collects the model data from a source webpage. The
    metadata will be stored within a dataframe and saved locally. If there
    is no metadata (i.e., trained models), the function returns None."""

    # Make an HTTP GET request to download the .csv file
    response = requests.get(source_url)

    soup = BeautifulSoup(response.text, "html.parser")

    script = None
    # Of the scripts, find the one containing model metadata
    for script in soup.find_all("script"):
        if "Classifier overview" in script.text:
            script = script.text
            break

    # If the page doesn't have model metadata, then return None early
    if script is None:
        return script

    # Parse the data when represented as a string, convert to dict
    data = script[script.find("data:"):].split(",\n")[0]
    data = eval(data[data.find("[{"):])

    # Loop through and collect meta from each model instance, store in list
    meta = []
    for point in data:

        score = point["y"]
        nimages = point["nimages"]
        traintime = point["traintime"]
        date = point["date"]
        source_id = point["pk"]

        meta.append([score, nimages, traintime, date, source_id])

    # Convert list to dataframe
    df = pd.DataFrame(meta, columns=['Accuracy %', 'N_Images',
                                     'Train_Time', 'Date', 'Source_ID'])
    # Save locally
    df.to_csv(source_dir + "model_metadata.csv")

    if os.path.exists(source_dir + "model_metadata.csv"):
        print("Annotations saved to: ", source_dir + "model_metadata.csv")
    else:
        print("Could not download model metadata.")

    return df


def download_annotations(image_url, source_dir, anno_dir):
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
        response = session.get(image_url,
                               data=data,
                               headers=headers,
                               cookies=cookies)

        # Pass along the cookies
        cookies = response.cookies

        # Parse the HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the form in the HTML
        form = soup.find("form", {"id": "export-annotations-form"})

        # If form comes back empty, it's likely the credentials are incorrect
        if form is None:
            print("Annotations could not be downloaded; it looks like the "
                  "CoralNet Username and Password are incorrect!")
            return

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
            df.to_csv(source_dir + "all_annotations.csv")

            # Save annotations per image
            for image_name in df['Name'].unique():
                image_annotations = df[df['Name'] == image_name]
                # Save in annotation folder
                anno_name = image_name.split(".")[0] + ".csv"
                image_annotations.to_csv(anno_dir + anno_name)

        else:
            print("Could not connect with CoralNet; please ensure that you "
                  "entered your username, password, and source ID correctly.")

    if os.path.exists(source_dir + "all_annotations.csv"):
        print("Annotations saved to: ", source_dir + "all_annotations.csv")
    else:
        print("Could not download annotations.")


def download_image(row, image_dir):
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
    image_path = image_dir + row[0]

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


def download_images(image_url, source_dir, image_dir):
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
    df.to_csv(source_dir + str(source_dir) + "_images.csv")

    # Loop through all the URLs, and download each image
    [download_image(row, image_dir) for row in df.values.tolist()]


def download_data(source_id):
    """This function serves as the front for downloading all the data
    (labelset, model metadata, annotations and images) for a source. This
    function was made so that multiprocessing can be used to download the
    data for multiple sources concurrently."""

    # Constant URLs for getting images, labelset, and annotations
    source_url = CORALNET_URL + "/source/" + str(source_id)
    image_url = source_url + "/browse/images/"
    labelset_url = source_url + "/labelset/"

    # The directory to store the output
    source_dir = OUTPUT_DIR + "/" + str(source_id) + "/"
    image_dir = source_dir + "images/"
    anno_dir = source_dir + "annotations/"

    # Creating the directories
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    # Download the metadata of the trained model, if exists
    download_model_meta(source_url, source_dir)

    # Download the label set as a csv
    download_labelset(labelset_url, source_dir)

    # Download all the images
    download_images(image_url)

    # Check to see if user credentials have been set,
    # if not, annotations cannot be downloaded; skip.
    if not None in [USERNAME, PASSWORD]:
        # Download the annotations as a csv
        download_annotations(image_url, source_dir, anno_dir)

def parse_labelset_file(path):
    """Helper function to extract the labelsets from a provided file.
    Returns a list containing all labelsets."""

    labelset = []

    # Checks that file exists
    if os.path.exists(path):
        try:
            # Attempts to open file in read mode, parses, stores in list
            with open(path, "r") as f:
                for line in f:
                    items = line.strip().split(',')[0]
                    labelset.append(items)

        except:
            print("Could not parse file correct. Please ensure that each "
                  "labelset code is on it's own line, followed by a comma.")

    else:
        print("Path does not exist: ", path)
        print("Exiting.")
        sys.exit()

    return labelset


def download_coralnet_sources():
    """Downloads a list of all the public sources currently on CoralNet."""

    print("Downloading CoralNet Source List...")

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
        response = session.get(CORALNET_SOURCE_URL)

        # Pass along the cookies
        cookies = response.cookies

        # Parse the HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all('ul', class_='object_list')[0].find_all("li")

        source_ids = []
        # Now, get all the source id's on the page
        for link in links:
            href = link.find("a").get("href")
            source_id = href.split("/")[-2]
            # Confirm that the value is numerical
            if source_id.isnumeric():
                source_ids.append(source_id)

        df = pd.DataFrame(source_ids, columns=['Source_ID'])
        df.to_csv("CoralNet_Source_ID_List.csv")

        if os.path.exists("CoralNet_Source_ID_List.csv"):
            print("Source ID list exported successfully.")
        else:
            print("Could not download Source ID list; check that variable "
                  "CORALNET_SOURCE_URL is correct.")

    return df


def download_coralnet_labelset():
    """Downloads a list of all the Labelsets currently on CoralNet."""

    print("Downloading CoralNet Labeset List...")

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
        response = session.get(CORALNET_LABELSET_URL)

        # Pass along the cookies
        cookies = response.cookies

        # Parse the HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Get the table with all labelset information
        table = soup.find_all('tr', attrs={"data-label-id": True})

        # Loop through each row, grab the information, store in lists
        rows = []
        for row in table:
            # Grab attributes from row
            attributes = row.find_all("td")
            # Extract each attribute, store in variable
            name = attributes[0].text
            url = CORALNET_URL + attributes[0].find("a").get("href")
            functional_group = attributes[1].text
            popularity = attributes[2].find("div").get("title").split("%")[0]
            short_code = attributes[4].text

            is_duplicate = False
            is_verified = False
            has_calcification = False
            notes = ""
            # Loop through the optional attributes
            for column in attributes[3].find_all("img"):
                if column.get("alt") == "Duplicate":
                    is_duplicate = True
                    notes = column.get("title")
                if column.get("alt") == "Verified":
                    is_verified = True
                if column.get("alt") == "Has calcification rate data":
                    has_calcification = True

            rows.append([name, url, functional_group, popularity, short_code,
                         is_duplicate, notes, is_verified, has_calcification])

        # Create dataframe
        df = pd.DataFrame(rows, columns=['Name', 
                                         'URL', 
                                         'Functional_Group',
                                         'Popularity %', 
                                         'Short_Code',
                                         'Duplicate', 
                                         'Duplicate Notes',
                                         'Verified',
                                         'Has_Calcification_Rates'])

        # Save locally
        df.to_csv("CoralNet_Labelset_List.csv")

        if os.path.exists("CoralNet_Labelset_List.csv"):
            print("Labelset list exported successfully.")
        else:
            print("Could not download Labelset list; check that variable "
                  "CORALNET_LABELSET_URL is correct.")

    return df


def get_source_ids(labelsets):
    """This function takes in a pandas dataframe containing a subset of
    labelsets and returns a list of source ids that contain those labelsets."""

    # List of source ids that contain the desired labelset(s)
    source_ids = []

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

        # Loop through all labelset URLs
        for url in labelsets['URL'].values:

            # Use session.get() to make a GET request to the source URL
            response = session.get(url)

            # Pass along the cookies
            cookies = response.cookies

            # Parse the HTML response using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all the source ids on the page
            a_tags = soup.find_all('a')
            a_tags = [a for a in a_tags if '/source/' in a.get('href')]
            source_id = [a.get("href").split("/")[-2] for a in a_tags]
            source_id = [id for id in source_id if id.isnumeric()]

            # Extend the list, remove duplicates
            source_ids.extend(source_id)
            source_ids = [*set(source_ids)]

            # If the list of source ids is not empty, save locally
            if source_ids is not []:
                with open('Desired_Source_ID_List.txt', 'w') as f:
                    f.write(','.join(str(_) for _ in source_ids))

        if os.path.exists("Desired_Source_ID_List.txt"):
            print("Source ID List saved successfully.")
        else:
            print("Could not find / download list of Source IDs.")

    return source_ids


# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='CoralNet arguments')

parser.add_argument('--username', type=str, default=os.getenv('CORALNET_USERNAME'),
                    help='Username for CoralNet account')

parser.add_argument('--password', type=str, default=os.getenv('CORALNET_PASSWORD'),
                    help='Password for CoralNet account')

parser.add_argument('--source_ids', type=int, nargs='+',
                    help='A list of source IDs to download.')

parser.add_argument('--output_dir', type=str, default="download",
                    help='A root directory where all downloads will be saved to.')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Constant for the CoralNet url
CORALNET_URL = "https://coralnet.ucsd.edu"

# CoralNet Source page, lists all sources
CORALNET_SOURCE_URL = CORALNET_URL + "/source/about/"

# CoralNet Labelset page, lists all labelsets
CORALNET_LABELSET_URL = CORALNET_URL + "/label/list/"

# URL of the login page
LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"

# A list of sources to download
SOURCE_IDS = args.source_ids

# Credentials
USERNAME = args.username
PASSWORD = args.password

# Output directory
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    """This is the main function of the script. It calls the functions 
    download_labelset, download_annotations, and download_images to download 
    the label set, annotations, and images, respectively.
    
    There are other functions that also allow you to identify all public
    sources, all labelsets, and sources containing specific labelsets.
    It is entirely possibly to identify sources based on labelsets, and
    download all those sources, or simply download all data from all
    source. Have fun!"""

    # Create a `Pool` object with the number of processes you have, minus 1
    pool = multiprocessing.Pool(processes=os.cpu_count()-1)

    # Apply the `download_data` function to every element in `SOURCE_IDs`
    # in parallel
    pool.map(download_data, SOURCE_IDS)

    print("Done.")









