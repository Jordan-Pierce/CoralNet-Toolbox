import gradio as gr

from common import *

from Tools.API import api

EXIT_APP = False
log_file = "annotate.log"


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(username, password, source_id_1, source_id_2, points, prefix, output_dir):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(log_file)

    args = argparse.Namespace(
        username=username,
        password=password,
        source_id_1=source_id_1,
        source_id_2=source_id_2,
        points=points,
        prefix=prefix,
        output_dir=output_dir,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        api(args)
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

    with gr.Blocks(title="CoralNet API 🕹️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# CoralNet API 🕹️")

        # Input Parameters
        with gr.Row():
            username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
            password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')

        with gr.Row():
            source_id_1 = gr.Textbox("", label="Source ID (for images)")
            source_id_2 = gr.Textbox("", label="Source ID (for model)")
            prefix = gr.Textbox("", label="Image Name Prefix")

        # Files button
        points = gr.Textbox("", label="Selected Points File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_files, outputs=points, show_progress="hidden")

        # Browse button
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [username,
                                    password,
                                    source_id_1,
                                    source_id_2,
                                    points,
                                    prefix,
                                    output_dir])

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