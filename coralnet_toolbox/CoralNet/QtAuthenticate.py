import os
import requests
import ujson as json
from bs4 import BeautifulSoup

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox, QGroupBox,
                             QFormLayout, QApplication)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


# Constant for the CoralNet url
CORALNET_URL = "https://coralnet.ucsd.edu"

# CoralNet Source page, lists all sources
CORALNET_SOURCE_URL = f"{CORALNET_URL}/source/about/"

# CoralNet Labelset page, lists all labelsets
CORALNET_LABELSET_URL = f"{CORALNET_URL}/label/list/"

# URL of the login page
LOGIN_URL = f"{CORALNET_URL}/accounts/login/"


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AuthenticateDialog(QDialog):
    """
    QDialog for authenticating with CoralNet.
    Provides UI for username/password input and manages the authentication token.
    """

    def __init__(self, parent=None):
        super(AuthenticateDialog, self).__init__(parent)

        self.token = None
        self.headers = None
        self.authenticated = False

        # Class constants for URLs
        self.CORALNET_URL = CORALNET_URL
        self.CORALNET_SOURCE_URL = CORALNET_SOURCE_URL
        self.CORALNET_LABELSET_URL = CORALNET_LABELSET_URL
        self.LOGIN_URL = LOGIN_URL

        self.setWindowTitle("Authenticate")
        self.resize(600, 200)  # Width, height

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the login layout
        self.setup_login_layout()
        # Setup the token display layout
        self.setup_token_layout()
        # Setup buttons layout
        self.setup_buttons_layout()

        # Load saved credentials if available
        self.load_saved_credentials()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Authenticate with CoralNet using your usename and password.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_login_layout(self):
        """Set up the username and password input section of the UI."""
        # Create a group box for login credentials
        login_group = QGroupBox("Login Credentials")
        form_layout = QFormLayout()

        # Username input
        self.username_input = QLineEdit()
        form_layout.addRow("Username:", self.username_input)

        # Password input with toggle button
        password_layout = QHBoxLayout()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        password_layout.addWidget(self.password_input)

        # Add toggle button for password
        self.toggle_password_button = QPushButton("Show")
        self.toggle_password_button.setFixedWidth(50)
        self.toggle_password_button.clicked.connect(self.toggle_password_visibility)
        password_layout.addWidget(self.toggle_password_button)

        form_layout.addRow("Password:", password_layout)

        # Set the form layout to the group box
        login_group.setLayout(form_layout)

        # Add the group box to the main layout
        self.layout.addWidget(login_group)

    def setup_token_layout(self):
        """Set up the information display section of the UI."""
        # Create a group box for token information
        token_group = QGroupBox("Authentication Status")
        form_layout = QFormLayout()

        # Token display with toggle button
        token_layout = QHBoxLayout()
        self.token_display = QLineEdit()
        self.token_display.setReadOnly(True)  # Makes the field read-only but still selectable/copyable
        token_layout.addWidget(self.token_display)

        # Add toggle button for token
        self.toggle_token_button = QPushButton("Show")
        self.toggle_token_button.setFixedWidth(50)
        self.toggle_token_button.clicked.connect(self.toggle_token_visibility)
        token_layout.addWidget(self.toggle_token_button)

        form_layout.addRow("Authentication Token:", token_layout)

        # Status information in a read-only QLineEdit
        self.status_display = QLineEdit("Not authenticated")
        self.status_display.setReadOnly(True)  # Makes the field read-only but still selectable/copyable
        form_layout.addRow("Logged in as:", self.status_display)

        # Set the form layout to the group box
        token_group.setLayout(form_layout)

        # Add the group box to the main layout
        self.layout.addWidget(token_group)

    def setup_buttons_layout(self):
        """Set up the login, logout and exit buttons section of the UI."""
        button_layout = QHBoxLayout()

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.login)

        self.logout_button = QPushButton("Logout")
        self.logout_button.clicked.connect(self.logout)
        self.logout_button.setEnabled(False)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.reject)

        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.logout_button)
        button_layout.addWidget(self.exit_button)

        # Add to main layout
        self.layout.addLayout(button_layout)

    def toggle_password_visibility(self):
        """Toggle password field visibility between visible and hidden."""
        if self.password_input.echoMode() == QLineEdit.Password:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_button.setText("Hide")
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_button.setText("Show")

    def toggle_token_visibility(self):
        """Toggle token field visibility between visible and hidden."""
        if self.token_display.echoMode() == QLineEdit.Password:
            self.token_display.setEchoMode(QLineEdit.Normal)
            self.toggle_token_button.setText("Hide")
        else:
            self.token_display.setEchoMode(QLineEdit.Password)
            self.toggle_token_button.setText("Show")

    def login(self):
        """Authenticate with CoralNet and retrieve token."""
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self,
                                "Authentication Error",
                                "Please enter both username and password")
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # First authenticate using requests
            self.authenticate(username, password)

            # Then get the token
            self.token, self.headers = self.get_token(username, password)

            # Store credentials in environment variables
            os.environ['CORALNET_USERNAME'] = username
            os.environ['CORALNET_PASSWORD'] = password

            # Update UI
            self.token_display.setText(self.token)
            self.token_display.setEchoMode(QLineEdit.Password)  # Start in hidden mode
            self.toggle_token_button.setText("Show")
            self.status_display.setText(f"{username}")
            self.authenticated = True

            # Toggle button states
            self.login_button.setEnabled(False)
            self.logout_button.setEnabled(True)

            # Make fields read-only while logged in
            self.username_input.setReadOnly(True)
            self.password_input.setReadOnly(True)

            # Show success message
            QMessageBox.information(self,
                                    "Authentication Successful",
                                    "You are now authenticated!")

        except Exception as e:
            QMessageBox.critical(self, "Authentication Error", str(e))

        finally:
            # Reset cursor
            QApplication.restoreOverrideCursor()

    def logout(self):
        """Log out and clear authentication information."""
        self.token = None
        self.headers = None
        self.authenticated = False

        # Update UI
        self.token_display.clear()
        self.token_display.setEchoMode(QLineEdit.Normal)  # Reset to normal mode
        self.toggle_token_button.setText("Show")
        self.status_display.setText("Not authenticated")

        # Toggle button states
        self.login_button.setEnabled(True)
        self.logout_button.setEnabled(False)

        # Make fields editable again
        self.username_input.setReadOnly(False)
        self.password_input.setReadOnly(False)

    def authenticate(self, username, password):
        """
        Authenticate with CoralNet; used to make sure user has the correct credentials.
        """
        try:
            # Send a GET request to the login page to retrieve the login form
            response = requests.get(self.LOGIN_URL, timeout=30)
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
            "username": username,
            "password": password,
            "csrfmiddlewaretoken": csrf_token["value"],
        }

        # Include the "Referer" header in the request
        headers = {
            "Referer": self.LOGIN_URL,
        }

        # Use requests.Session to create a session that will maintain your login state
        with requests.Session() as session:
            # Use session.post() to submit the login form
            response = session.post(self.LOGIN_URL,
                                    data=data,
                                    headers=headers,
                                    cookies=cookies)

            if "credentials you entered did not match" in response.text:
                raise Exception(f"Authentication unsuccessful for '{username}'.\n"
                                f"Please check that your username and password are correct")

    def get_token(self, username, password):
        """
        Retrieves a CoralNet authentication token for API requests.
        """
        # Requirements for authentication
        CORALNET_AUTH = self.CORALNET_URL + "/api/token_auth/"
        HEADERS = {"Content-type": "application/vnd.api+json"}
        PAYLOAD = {"username": username, "password": password}

        try:
            # Response from CoralNet when provided credentials
            response = requests.post(CORALNET_AUTH,
                                     data=json.dumps(PAYLOAD),
                                     headers=HEADERS,
                                     timeout=10)
        except Exception as e:
            raise Exception(f"ERROR: {e}")

        if response.ok:
            # Get the coralnet token returned to the user
            CORALNET_TOKEN = json.loads(response.content.decode())['token']

            # Update the header to contain the user's coralnet token
            HEADERS = {"Authorization": f"Token {CORALNET_TOKEN}",
                       "Content-type": "application/vnd.api+json"}
        else:
            raise ValueError(f"ERROR: Could not retrieve API token\n{response.content}")

        return CORALNET_TOKEN, HEADERS

    def get_auth_token(self):
        """
        Returns the current authentication token.
        """
        return self.token

    def get_auth_headers(self):
        """
        Returns the headers with authentication token.
        """
        return self.headers

    def is_authenticated(self):
        """
        Returns True if the user is authenticated, False otherwise.
        """
        return self.authenticated

    def load_saved_credentials(self):
        """Load saved credentials from environment variables if available."""
        username = os.environ.get('CORALNET_USERNAME', '')
        password = os.environ.get('CORALNET_PASSWORD', '')

        if username:
            self.username_input.setText(username)
        if password:
            self.password_input.setText(password)
