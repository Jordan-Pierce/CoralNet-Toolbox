import gradio as gr

from Toolbox.Pages.common import *

from Toolbox.Tools.SAM import sam

EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(images, annotations, label_col, confidence, model_type, points_per_side, points_per_batch,
                    output_dir):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        images=images,
        annotations=annotations,
        label_col=label_col,
        confidence=confidence,
        model_type=model_type,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        output_dir=output_dir,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        sam(args)
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

    with gr.Blocks(title="SAM 🧠", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# SAM 🧠")

        # Browse button
        images = gr.Textbox(f"{DATA_DIR}", label="Selected Image Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=images, show_progress="hidden")

        annotations = gr.Textbox(label="Selected Annotation File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_file, outputs=annotations, show_progress="hidden")

        with gr.Row():

            label_col = gr.Dropdown(label="Label Name Field", multiselect=False, allow_custom_value=True,
                                    choices=['Label'] + [f'Machine suggestion {n + 1}' for n in range(5)])

            confidence = gr.Slider(label="Point Confidence Filter",
                                   minimum=0, maximum=100, step=1)

        with gr.Group("SAM Model"):
            with gr.Row():

                model_type = gr.Dropdown(label="SAM Model Weights",
                                         choices=['vit_b', 'vit_l', 'vit_h'],
                                         multiselect=False, allow_custom_value=False)

                points_per_side = gr.Number(64, label="Number of Points (Squared)",
                                            precision=0)

                points_per_batch = gr.Number(128, label="Points per Batch (GPU dependent)",
                                             precision=0)

        # Browse button
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [images,
                                    annotations,
                                    label_col,
                                    confidence,
                                    model_type,
                                    points_per_side,
                                    points_per_batch,
                                    output_dir])

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