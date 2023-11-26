import gradio as gr

from Toolbox.Pages.common import *

from Toolbox.Tools.SfM import sfm

EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(metashape_license, input_dir, output_dir, project_file, quality, target_percentage):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        metashape_license=metashape_license,
        input_dir=input_dir,
        output_dir=output_dir,
        project_file=project_file,
        quality=quality,
        target_percentage=target_percentage,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        sfm(args)
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

    with gr.Blocks(title="SfM (Metashape) 🧊", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# SfM (Metashape) 🧊")

        metashape_license = gr.Textbox(os.getenv('METASHAPE_LICENSE'),
                                       label="Metashape License", type='password')

        with gr.Tab(label="New Project"):

            # Browse button
            input_dir = gr.Textbox(label="Selected Image Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=input_dir, show_progress="hidden")

            output_dir = gr.Textbox(label="Selected Output Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Tab(label="Existing Project"):

            project_file = gr.Textbox(label="Selected Project File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=project_file, show_progress="hidden")

        with gr.Row():

            quality = gr.Dropdown(label="Reconstruction Quality",
                                        choices=['Lowest', 'Low', 'Medium', 'High', 'Highest'],
                                        multiselect=False)

            target_percentage = gr.Slider(label="Gradual Selection Percentage", interactive=True,
                                          minimum=0, maximum=100, step=5)

        # Browse button

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [metashape_license,
                                    input_dir,
                                    output_dir,
                                    project_file,
                                    quality,
                                    target_percentage])

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

