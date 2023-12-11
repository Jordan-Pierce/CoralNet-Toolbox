import gradio as gr

from Toolbox.Pages.common import *

from Toolbox.Tools.Classification import get_classifier_encoders
from Toolbox.Tools.ClassificationPreTrain import classification_pretrain

EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(patches, encoder_name, freeze_encoder, projection_dim, optimizer, num_epochs, batch_size,
                    tensorboard, output_dir):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(LOG_PATH)

    # Custom pre-processing
    patches = patches.split(" ")

    args = argparse.Namespace(
        patches=patches,
        encoder_name=encoder_name,
        freeze_encoder=freeze_encoder,
        projection_dim=projection_dim,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        tensorboard=tensorboard,
        output_dir=output_dir,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        classification_pretrain(args)
        print("\nDone.")
        gr.Info("Completed process!")
    except Exception as e:
        gr.Error("Could not complete process!")
        print(f"ERROR: {e}\n{traceback.format_exc()}")

    sys.stdout = console


def tensorboard_iframe():
    """

    """
    url = 'http://localhost:6006/#timeseries'
    iframe = """<iframe src="{}" style="width:100%; height:1000px;"></iframe>""".format(url)
    return iframe


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

    with gr.Blocks(title="Pre-Train 👩‍🏫", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("Pre-Train 👩‍🏫")

        # Browse button
        patches = gr.Textbox(label="Selected Patches File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_files, outputs=patches, show_progress="hidden")

        with gr.Group("Training Parameters"):
            #
            with gr.Row():
                encoder_name = gr.Dropdown(label="Encoder", multiselect=False, allow_custom_value=False,
                                           choices=get_classifier_encoders())

                freeze_encoder = gr.Slider(0.0, label="Freeze Encoder", minimum=0.0, maximum=1.0, step=0.01)

            with gr.Row():

                projection_dim = gr.Number(64, label="Projection Dimensions", precision=0)

                optimizer = gr.Dropdown(label="Optimizer", multiselect=False, allow_custom_value=False,
                                        choices=['Adam, LARS'])

            with gr.Row():
                num_epochs = gr.Number(25, label="Number of Epochs", precision=0)

                batch_size = gr.Number(128, label="Batch Size (Power of 2 Recommended)", precision=0)

                tensorboard = gr.Dropdown(label="Tensorboard", multiselect=False, allow_custom_value=False,
                                          choices=[True, False])

        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [patches,
                                    encoder_name,
                                    freeze_encoder,
                                    projection_dim,
                                    optimizer,
                                    num_epochs,
                                    batch_size,
                                    tensorboard,
                                    output_dir])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(exit_interface)

        with gr.Accordion("Console Logs"):
            # Add to console in page
            logs = gr.Code(label="", language="shell", interactive=False, container=True, lines=30)
            interface.load(read_logs, None, logs, every=1)

        with gr.Accordion("TensorBoard"):
            # Display Tensorboard in page
            tensorboard_button = gr.Button("Show TensorBoard")
            iframe = gr.HTML(every=0.1)
            tensorboard_button.click(fn=tensorboard_iframe, outputs=iframe, every=0.1)

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