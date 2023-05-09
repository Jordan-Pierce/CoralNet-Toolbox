import os
import json
import requests
from bs4 import BeautifulSoup

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

# -----------------------------------------------------------------------------
# Functions to authenticate with CoralNet
# -----------------------------------------------------------------------------

def authenticate(username, password):

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
        "username": username,
        "password": password,
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

        if "credentials you entered did not match" in response.text:
            raise Exception("Login failed. Please check your username and "
                            "password.")
        else:
            print(f"NOTE: Successfully logged in for {username}")


def get_token(username, password):
    """
    Retrieves a CoralNet authentication token for API requests.

    Returns:
        tuple: A tuple containing the CoralNet token and request headers for
        authenticated requests.

    Raises:
        ValueError: If authentication fails.
    """

    # Requirements for authentication
    CORALNET_AUTH = CORALNET_URL + "/api/token_auth/"
    HEADERS = {"Content-type" : "application/vnd.api+json"}
    PAYLOAD = {"username": username, "password": password}

    # Response from CoralNet when provided credentials
    response = requests.post(CORALNET_AUTH,
                             data=json.dumps(PAYLOAD),
                             headers=HEADERS)

    if response.ok:

        print("NOTE: Successful authentication")

        # Get the coralnet token returned to the user
        CORALNET_TOKEN = json.loads(response.content.decode())['token']

        # Update the header to contain the user's coralnet token
        HEADERS = {"Authorization": f"Token {CORALNET_TOKEN}",
                   "Content-type": "application/vnd.api+json"}

    else:
        raise ValueError(f"ERROR: Could not authenticate\n{response.content}")

    return CORALNET_TOKEN, HEADERS