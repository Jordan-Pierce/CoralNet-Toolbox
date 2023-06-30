from CoralNet_API import *


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------


def cleanup():
    """Removes temp file to avoid clutter"""
    file = "gooey_config.json"
    if os.path.exists(file):
        # Remove the config file
        os.remove("gooey_config.json")


@Gooey(dump_build_config=True,
       program_name="CoralNet API",
       default_size=(800, 600),
       console=True,
       progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
       progress_expr="current / total * 100",
       hide_progress_msg=True,
       timing_options={
           'show_time_remaining': True,
           'hide_time_remaining_on_complete': True,
       })
def main():
    desc = "Access trained CoralNet models for performing prediction"

    parser = GooeyParser(description=desc)

    # Panel 1 - Download by Source ID
    panel1 = parser.add_argument_group('CoralNet API',
                                       'Specify the Source by the ID, and provide one or more CSV '
                                       'file(s) containing the names of images to perform '
                                       'predictions on. Images must already exist in the Source, '
                                       'and CSV file(s) must contain the following fields: '
                                       'Name, Row, Column.')

    panel1.add_argument('--username', type=str,
                        metavar="Username",
                        default=os.getenv('CORALNET_USERNAME'),
                        help='Username for CoralNet account')

    panel1.add_argument('--password', type=str,
                        metavar="Password",
                        default=os.getenv('CORALNET_PASSWORD'),
                        help='Password for CoralNet account',
                        widget="PasswordField")

    panel1.add_argument('--remember_username', action="store_false",
                        metavar="Remember Username",
                        help='Store Username as an Environmental Variable',
                        widget="BlockCheckbox")

    panel1.add_argument('--remember_password', action="store_false",
                        metavar="Remember Password",
                        help='Store Password as an Environmental Variable',
                        widget="BlockCheckbox")

    panel1.add_argument('--source_id_1', type=str, required=True,
                        metavar="Source ID (for images)",
                        help='The ID of the Source containing images.')

    panel1.add_argument('--source_id_2', type=str, required=False,
                        metavar="Source ID (for model)",
                        default=None,
                        help='The ID of the Source containing the model to use, if different.')

    panel1.add_argument('--output_dir', required=True,
                        metavar='Output Directory',
                        default="..\\CoralNet_Data",
                        help='A root directory where all predictions will be saved to.',
                        widget="DirChooser")

    panel1.add_argument('--csv_path', required=True, type=str, nargs='+',
                        metavar="Annotation File(s)",
                        help='A path to one more csv files. Each csv file should '
                             'contain following: Name, Row, Column',
                        widget="MultiFileChooser", )

    args = parser.parse_args()

    # ----------------------------------------
    # Check the data
    # ----------------------------------------
    try:
        # To store the data to be annotated
        POINTS = pd.DataFrame()

        for file in args.csv_path:
            # This is a problem
            if not os.path.exists(file):
                raise Exception(f"ERROR: File not found {file}")

            POINTS = pd.concat([POINTS, pd.read_csv(file)])

        # Check to see if the csv file has the expected columns
        assert 'Name' in POINTS.columns
        assert 'Row' in POINTS.columns
        assert 'Column' in POINTS.columns
        assert len(POINTS) > 0

    except Exception as e:
        raise Exception(f"ERROR: File(s) provided do not match expected format!\n{e}")

    # ----------------------------------------
    # Authenticate the user
    # ----------------------------------------
    try:
        # Ensure the user provided a username and password.
        username = args.username
        password = args.password
        authenticate(username, password)
        CORALNET_TOKEN, HEADERS = get_token(username, password)

        # Double Negative; if user wants these stored, they get saved
        # as environmental variables, so in the future it will autofill.
        # This is done after authentication to ensure correct credentials.
        if not args.remember_username:
            os.environ['CORALNET_USERNAME'] = str(args.username)

        if not args.remember_password:
            os.environ['CORALNET_PASSWORD'] = str(args.password)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Get the browser
    # -------------------------------------------------------------------------
    driver = check_for_browsers(headless=True)
    # Store the credentials in the driver
    driver.capabilities['credentials'] = {
        'username': username,
        'password': password,
    }
    # Login to CoralNet
    driver, _ = login(driver)

    # ----------------------------------------
    # Get Source information
    # ----------------------------------------
    try:
        SOURCE_ID = args.source_id_1
        driver, meta, SOURCE_IMAGES = get_source_meta(driver, args.source_id_1, args.source_id_2)

        # Get the images desired for predictions; make sure it's not file path.
        images = POINTS['Name'].unique().tolist()
        images = [os.path.basename(image) for image in images]

        # We will get the information needed from the source images dataframe
        IMAGES = SOURCE_IMAGES[SOURCE_IMAGES['Name'].isin(images)].copy()
        print(f"NOTE: Found the {len(IMAGES)} images in the source {SOURCE_ID}")

        # Get the image AWS URLs for the images of interest
        image_pages = IMAGES['Image Page'].tolist()
        driver, IMAGES['Image URL'] = get_image_urls(driver, image_pages)

    except Exception as e:
        print(f"ERROR: Issue with getting Source Metadata.\n{e}")
        sys.exit()

    # Set the model ID and URL
    MODEL_ID = meta['Global id'].max()
    MODEL_URL = CORALNET_URL + f"/api/classifier/{MODEL_ID}/deploy/"

    # Set the data root directory
    DATA_ROOT = os.path.abspath(args.output_dir) + "\\"
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Where the output predictions will be stored
    SOURCE_DIR = DATA_ROOT + SOURCE_ID + "\\"
    PREDICTIONS_DIR = SOURCE_DIR + "predictions\\"

    # Create a folder to contain predictions and points
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # ---------------------------------------------------------------------------------------------
    # Make calls to the API
    # ---------------------------------------------------------------------------------------------
    api(driver, IMAGES, POINTS, MODEL_URL, CORALNET_TOKEN, HEADERS, PREDICTIONS_DIR)

    print("Done.")


if __name__ == '__main__':
    main()
    cleanup()
