import sys
import os.path

from CoralNet_Crawler import *

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


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
        df = pd.DataFrame(rows, columns=['Name', 'URL', 'Functional_Group',
                                         'Popularity %', 'Short_Code',
                                         'Duplicate', 'Duplicate Notes',
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


if __name__ == "__main__":
    """This is the main function of the script. It calls the functions 
    download_coralnet_sources and download_coralnet to download 
    the entire CoralNet Source and Labelset list, respectively. Then asks 
    the user if they want to download sources based on a set of desired 
    labels."""

    # Download the source list as a csv
    SOURCES = download_coralnet_sources()

    # Download the label set list as a csv
    LABELSETS = download_coralnet_labelset()

    print("\nFor reference, see this page: ", CORALNET_LABELSET_URL)

    # Get the list of desired labelsets
    mode = input("Choose how you'd like to select the labelsets: \n"
                 "1) Name\n"
                 "2) Functional Group\n"
                 "3) Short Code \n").lower()

    labelsets = input("Instructions: Enter the desired labelsets, followed "
                      "by a comma. \nNote: The Name, Functional Group, "
                      "or Short Code must match exactly as seen in the above "
                      "url. \nDesired Labelset: ")

    labelsets = [l.strip() for l in labelsets.split(",")]

    if mode in ['1', "name"]:
        labelsets = LABELSETS[LABELSETS['Name'].isin(labelsets)]

    elif mode in ['2', 'functional group', 'functional_group']:
        labelsets = LABELSETS[LABELSETS['Functional_Group'].isin(labelsets)]

    elif mode in ['3', 'short code', 'short_code']:
        labelsets = LABELSETS[LABELSETS['Short_Code'].isin(labelsets)]

    else:
        print("This mode is not a valid option: ", mode)
        print("Exiting.")

    print("Finding sources; See 'Desired_Source_ID_List.txt' for progress...")

    SOURCE_IDs = get_source_ids(labelsets)

    print("Done.")








