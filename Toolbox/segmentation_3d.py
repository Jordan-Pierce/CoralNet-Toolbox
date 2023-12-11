import gradio as gr

from common import *

from Tools.Segmentation3D import segmentation3d

EXIT_APP = False
log_file = "segmentation_3d.log"


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(metashape_license, project_file, masks_file, color_map, mask_column, binary_masks, chunk_index):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(log_file)

    args = argparse.Namespace(
        metashape_license=metashape_license,
        project_file=project_file,
        masks_file=masks_file,
        color_map=color_map,
        mask_column=mask_column,
        binary_masks=binary_masks,
        chunk_index=chunk_index
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        segmentation3d(args)
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

    with gr.Blocks(title="3D Semantic Segmentation 🤖️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# 3D Semantic Segmentation 🤖️️")

        metashape_license = gr.Textbox(os.getenv('METASHAPE_LICENSE'),
                                       label="Metashape License", type='password')

        project_file = gr.Textbox(label="Existing Project File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_file, outputs=project_file, show_progress="hidden")

        masks_file = gr.Textbox(label="Masks File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_file, outputs=masks_file, show_progress="hidden")

        color_map = gr.Textbox(label="Color Map File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_file, outputs=color_map, show_progress="hidden")

        with gr.Row():

            mask_column = gr.Dropdown(label="Mask Column", multiselect=False, allow_custom_value=True,
                                      choices=['Color Path', 'Overlay Path', 'Semantic Path'])

            chunk_index = gr.Number(0, label="Chunk Index", precision=0)

            binary_masks = gr.Checkbox(True, label="Include Binary Masks")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [metashape_license,
                                    project_file,
                                    masks_file,
                                    color_map,
                                    mask_column,
                                    binary_masks,
                                    chunk_index])

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