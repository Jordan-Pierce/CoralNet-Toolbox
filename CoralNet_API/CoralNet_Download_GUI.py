from CoralNet_Download import *


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
       program_name="CoralNet Download",
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
    """
    CoralNet Download Graphical User Interface

    This is the main function of the script. It calls the functions
    download_labelset, download_annotations, and download_images to download
    the label set, annotations, and images, respectively.

    There are other functions that also allow you to identify all public
    sources, all labelsets, and sources containing specific labelsets.
    It is entirely possibly to identify sources based on labelsets, and
    download all those sources, or simply download all data from all
    sources. Have fun!

    BE RESPONSIBLE WITH YOUR DOWNLOADS. DO NOT OVERWHELM THE SERVERS.
    """

    desc = "Download data from CoralNet"

    parser = GooeyParser(description=desc)

    # Panel 1 - Download by Source ID
    panel1 = parser.add_argument_group('Download By Source ID',
                                       'Specify each Source to download by providing the '
                                       'associated ID; for multiple sources, add a space between '
                                       'each.')

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

    panel1.add_argument('--source_ids', type=str, nargs='+',
                        metavar="Source IDs",
                        help='A list of source IDs to download (provide spaces between each ID).')

    panel1.add_argument('--output_dir', required=True,
                        metavar='Output Directory',
                        default="..\\CoralNet_Data",
                        help='A root directory where all downloads will be saved to.',
                        widget="DirChooser")

    panel1.add_argument('--headless', action="store_false",
                        metavar="Run in Background",
                        help='Run browser in headless mode',
                        widget='BlockCheckbox')

    # Panel 2 - Download CoralNet Source ID and Labelset Lists
    panel2 = parser.add_argument_group('Download CoralNet Dataframes',
                                       'In addition to downloading Source data, dataframes '
                                       'containing information on all public Sources and '
                                       'Labelsets can also be downloaded.')

    panel2.add_argument('--source_df', action="store_true",
                        metavar="Download Source Dataframe",
                        help='Information on all public Sources.',
                        widget='BlockCheckbox')

    panel2.add_argument('--labelset_df', action="store_true",
                        metavar="Download Labelset Dataframe",
                        help='Information on all Labelsets.',
                        widget='BlockCheckbox')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Authenticate the user
    # -------------------------------------------------------------------------
    try:
        # Ensure the user provided a username and password.
        username = args.username
        password = args.password
        authenticate(username, password)

        # Double Negative; if user wants these stored, they get saved
        # as environmental variables, so in the future it will autofill.
        # This is done after authentication to ensure correct credentials.
        if not args.remember_username:
            os.environ['CORALNET_USERNAME'] = str(args.username)

        if not args.remember_password:
            os.environ['CORALNET_PASSWORD'] = str(args.password)

    except Exception as e:
        print(f"ERROR: Could not download data.\n{e}")
        sys.exit(1)

    # Output directory
    output_dir = os.path.abspath(args.output_dir) + "\\"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Get the browser
    # -------------------------------------------------------------------------
    headless = not args.headless
    # Pass the options object while creating the driver
    driver = check_for_browsers(headless)
    # Store the credentials in the driver
    driver.capabilities['credentials'] = {
        'username': username,
        'password': password
    }
    # Login to CoralNet
    driver, _ = login(driver)

    # -------------------------------------------------------------------------
    # Download Dataframes
    # -------------------------------------------------------------------------
    if args.source_df and not os.path.exists(f"{output_dir}CoralNet_Source_ID_Dataframe.csv"):
        driver, source_df = download_coralnet_sources(driver, output_dir)

    if args.labelset_df and not os.path.exists(f"{output_dir}CoralNet_Labelset_Dataframe.csv"):
        driver, labelset_df = download_coralnet_labelsets(driver, output_dir)

    # -------------------------------------------------------------------------
    # Download Source Data
    # -------------------------------------------------------------------------
    if args.source_ids is not None:

        source_ids = args.source_ids
        source_ids = [s.strip() for s in source_ids]

        try:
            for idx, source_id in enumerate(source_ids):
                driver, m, l, i, a = download_data(driver, source_id, output_dir)
                print(f"progress: {idx + 1}/{len(source_ids)}")

        except Exception as e:
            raise Exception(f"ERROR: Could not download data\n{e}")

    # Close the browser
    driver.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
    cleanup()
