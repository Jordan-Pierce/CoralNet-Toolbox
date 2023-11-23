import gradio as gr

from Toolbox.Pages.common import *

from Toolbox.Tools.Projector import projector

EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(model, patches, output_dir, project_folder):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        model=model,
        patches=patches,
        output_dir=output_dir,
        project_folder=project_folder,
    )

    try:
        # Call the function
        projector(args)
        print("Done.")
    except Exception as e:
        gr.Error("Could not complete process")
        print(f"ERROR: {e}\n{traceback.format_exc()}")


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

    with gr.Blocks(title="Projector 📽️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Projector 📽️")

        with gr.Tab("New Project"):

            model = gr.Textbox(label="Selected Model File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=model, show_progress="hidden")

            patches = gr.Textbox(label="Selected Patches File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=patches, show_progress="hidden")

            # Browse button
            output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Tab("Existing Project"):

            # Browse button
            project_folder = gr.Textbox(label="Existing Project Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=project_folder, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [model,
                                    patches,
                                    output_dir,
                                    project_folder])

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

while True:
    time.sleep(0.5)
    if EXIT_APP:
        break
