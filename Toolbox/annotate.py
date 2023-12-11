import gradio as gr

from common import *

from Tools.Annotate import annotate

EXIT_APP = False
log_file = "annotate.log"


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(patch_extractor_path, image_dir):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(log_file)

    args = argparse.Namespace(
        patch_extractor_path=patch_extractor_path,
        image_dir=image_dir,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        annotate(args)
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

    with gr.Blocks(title="Annotate 🧮", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Annotate 🧮")

        # Browse button
        patch_extractor_path = gr.Textbox(f"{PATCH_EXTRACTOR}", label="Selected File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_file, outputs=patch_extractor_path, show_progress="hidden")

        # Browse button
        image_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Images Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=image_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [patch_extractor_path,
                                    image_dir])

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