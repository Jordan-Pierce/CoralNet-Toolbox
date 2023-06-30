from CoralNet import *


def generate_short_code(label, prefix, used_short_codes, num_chars=3):
    """
    Generate a descriptive short code for a given string, ensuring it is representative
    of the original string and distinct from any used short codes.
    """

    # Split the string into words
    tokens = label.split("_")
    short_code = ""

    # Generate the short code by taking the specified number of characters from each word
    for i, token in enumerate(tokens):
        short_code += token[:num_chars]

    # Truncate or pad the short code to ensure it is exactly 6 characters long
    short_code = short_code[:len(prefix)]

    # Convert the short code to uppercase
    short_code = prefix + short_code.upper()

    # Check if the generated short code is already used
    if used_short_codes and short_code in used_short_codes:
        # Add a number to the short code until it becomes unique
        count = 1
        new_short_code = short_code + str(count)

        while new_short_code in used_short_codes:
            count += 1
            new_short_code = short_code + str(count)

        short_code = new_short_code

    return short_code


def get_short_code(label, prefix, used_short_codes, num_chars=4):
    """
    Loops through generate short code to get a valid short code.
    """

    while True:
        # Keep generating short code until critera
        short_code = generate_short_code(label, prefix, used_short_codes, num_chars)
        # Check if it's too long, regenerate
        if not len(short_code) <= 10:
            num_chars -= 1
        # Else break
        else:
            break

    # Throw an error that it's too long
    if len(short_code) > 10:
        raise Exception(f"ERROR: Short code for {label} is too long {short_code}")

    return short_code


def create_labelset(driver, source_id, labelset):
    """
    Create a labelset given a labelset, which is a dict containing:
        Name - name of labelset
        Short Code - unique short code for label
        Functional Group - one of the pre-defined functional groups
        Description - text describing labelset
        Image Path - absolute path to image thumbnail

    Note: labelsets may be created for a source, but the labelset will be made public
    to all CoralNet users. Do not use this function frivolously.
    """

    # Create a variable to track the success of the upload
    success = False

    # First check that all the contents of labelset are valid
    try:
        # Character limitations, string for description
        assert len(str(labelset['Name'])) <= 45
        assert len(str(labelset['Short Code'])) <= 10
        assert str(labelset['Functional Group']) in FUNC_GROUPS_LIST
        assert str(labelset['Description']) != ""

        # Ensure that the image path is absolute
        labelset['Image Path'] = os.path.abspath(labelset['Image Path'])
        assert os.path.exists(labelset['Image Path'])
        assert labelset['Image Path'].split(".")[-1] in IMG_FORMATS

    except Exception as e:
        print(f"ERROR: Labelset provided does not match criteria; exiting")
        return driver, success

    print("\nNOTE: Navigating to labelset creation page")

    # Go to the upload page
    driver.get(CORALNET_URL + f"/source/{source_id}/labelset/add/")

    # First check that this is existing source the user has access to
    driver, success = check_permissions(driver)

    # If the user does not have access to the source, exit immediately
    if not success:
        print("ERROR: Cannot continue with process; exiting function.")
        return driver, success

    try:
        # Wait for the presence of the button
        path = "new-label-form-show-button"
        button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Click the button if it's present
        if button.is_displayed():
            button.click()
        else:
            raise Exception("ERROR: Create Labelset button not enabled")

        # Wait for the presence of the input field
        path = "id_name"
        input_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Send a string to the input field
        input_field.clear()
        input_field.send_keys(labelset['Name'])

        # Wait for the presence of the input field
        path = "id_default_code"
        input_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Send a string to the input field
        input_field.clear()
        input_field.send_keys(labelset['Short Code'])

        # Wait for the presence of the select element
        path = "id_group"
        select_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Select the "Other" option
        select = Select(select_element)
        select.select_by_value(FUNC_GROUPS_DICT[labelset['Functional Group']])

        # Wait for the presence of the textarea field
        path = "id_description"
        textarea_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        # Send a string to the textarea field
        textarea_field.clear()
        textarea_field.send_keys(labelset['Description'])

        # Wait for the presence of the select element
        path = "id_thumbnail"
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, path)))

        try:
            # Set the local file path to the file input field
            file_input.send_keys(labelset['Image Path'])
        except Exception as e:
            raise Exception(f"ERROR: Could not submit {labelset['Image Path']}")

        # Wait for the presence of the submit button
        path = "//input[@type='submit' and @value='Create Label']"
        submit_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, path)))

        # Click the submit button if it's present
        if submit_button.is_displayed():
            submit_button.click()
            print(f"NOTE: Submitted labelset {labelset['Name']}, {labelset['Short Code']}")
            success = True
        else:
            raise Exception("ERROR: Submit Labelset button not enabled")

    except Exception as e:
        print(f"ERROR: Could not create labelset {labelset['Name']}\n{e}")

    return driver, success