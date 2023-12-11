import gradio as gr

from common import *

from Tools.Labelset import labelset

EXIT_APP = False
log_file = "labelset.log"


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(username, password, source_id, labelset_name, short_code, func_group, desc, image_path, headless):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(log_file)

    args = argparse.Namespace(
        username=username,
        password=password,
        source_id=source_id,
        labelset_name=labelset_name,
        short_code=short_code,
        func_group=func_group,
        desc=desc,
        image_path=image_path,
        headless=headless
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        labelset(args)
        print("\nDone.")
        gr.Info("Completed process!")
    except Exception as e:
        gr.Error("Could not complete process!")
        print(f"ERROR: {e}\n{traceback.format_exc()}")

    sys.stdout = console


# ----------------------------------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------------------------------
def exit_interface():
    """

    """
    global EXIT_APP
    EXIT_APP = True

    gr.Info("Please close the browser tab.")
    gr.Info("Stopped program successfully!")
    time.sleep(3)


def create_interface():
    """

    """
    logger = Logger(log_file)
    logger.reset_logs()

    with gr.Blocks(title="Labelset 📝", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Labelset 📝")

        with gr.Row():
            username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
            password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')

        with gr.Row():
            source_id = gr.Number(label="Source ID", precision=0)
            headless = gr.Checkbox(label="Run Browser in Headless Mode", value=True)

        with gr.Row():
            labelset_name = gr.Textbox(label="Labelset Name")
            short_code = gr.Textbox(label="Short Code")
            func_group = gr.Dropdown(label="Functional Group", multiselect=False,
                                     choices=FUNC_GROUPS_LIST)

        desc = gr.Textbox(label="Labelset Description")

        image_path = gr.Textbox(f"{DATA_DIR}", label="Selected Thumbnail File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_file, outputs=image_path, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [username,
                                    password,
                                    source_id,
                                    labelset_name,
                                    short_code,
                                    func_group,
                                    desc,
                                    image_path,
                                    headless])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(exit_interface)

        with gr.Accordion("Console Logs"):
            # Add logs
            logs = gr.Code(label="", language="shell", interactive=False, container=True, lines=30)
            interface.load(logger.read_logs, None, logs, every=1)

    interface.launch(prevent_thread_lock=True, server_port=get_port(), inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
interface = create_interface()

try:
    while True:
        time.sleep(0.5)
        if EXIT_APP:
            break
except:
    pass

finally:
    Logger(log_file).reset_logs()