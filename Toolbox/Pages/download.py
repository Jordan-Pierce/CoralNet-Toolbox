import gradio as gr

from Toolbox.Pages.common import *

from Toolbox.Tools.Download import download
from Toolbox.Tools.Download import get_updated_labelset_list

EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(username, password, source_ids, source_df, labelset_df, sources_with, output_dir, headless):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(LOG_PATH)

    # Custom pre-processing
    source_ids = [str(id.strip()) for id in source_ids.split(" ") if id.strip()]

    args = argparse.Namespace(
        username=username,
        password=password,
        source_ids=source_ids,
        source_df=source_df,
        labelset_df=labelset_df,
        sources_with=sources_with,
        output_dir=output_dir,
        headless=headless
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        download(args)
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
    Logger(LOG_PATH).reset_logs()

    with gr.Blocks(title="CoralNet Download ⬇️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# CoralNet Downloader ⬇️")

        # Input Parameters
        with gr.Tab("Download Source Data"):
            with gr.Row():
                username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
                password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')

            with gr.Row():
                source_ids = gr.Textbox("", label="Source IDs (space-separated)")
                headless = gr.Checkbox(label="Run Browser in Headless Mode", value=True)

        with gr.Tab("Download CoralNet Dataframes"):
            with gr.Row():
                source_df = gr.Checkbox(label="Download Source DataFrame")
                labelset_df = gr.Checkbox(label="Download Labelset DataFrame")

            sources_with = gr.Dropdown(label="Sources with Labelsets",
                                       choices=get_updated_labelset_list(),
                                       multiselect=True)

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
                                    source_ids,
                                    source_df,
                                    labelset_df,
                                    sources_with,
                                    output_dir,
                                    headless])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(exit_interface)

        with gr.Accordion("Console Logs"):
            # Add logs
            logs = gr.Code(label="", language="shell", interactive=False, container=True, lines=30)
            interface.load(read_logs, None, logs, every=1)

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
    Logger(LOG_PATH).reset_logs()


