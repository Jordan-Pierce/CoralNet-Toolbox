import gradio as gr

from common import *

from Tools.Segmentation import segmentation
from Tools.Segmentation import get_segmentation_encoders
from Tools.Segmentation import get_segmentation_decoders
from Tools.Segmentation import get_segmentation_metrics
from Tools.Segmentation import get_segmentation_losses
from Tools.Segmentation import get_segmentation_optimizers

EXIT_APP = False
log_file = "segmentation.log"


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(masks, color_map, pre_trained_path, encoder_name, decoder_name, metrics, loss_function,
                    freeze_encoder, optimizer, learning_rate, augment_data, num_epochs, batch_size, tensorboard,
                    output_dir):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(log_file)

    args = argparse.Namespace(
        masks=masks,
        color_map=color_map,
        pre_trained_path=pre_trained_path,
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        metrics=metrics,
        loss_function=loss_function,
        freeze_encoder=freeze_encoder,
        optimizer=optimizer,
        learning_rate=learning_rate,
        augment_data=augment_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        tensorboard=tensorboard,
        output_dir=output_dir,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        segmentation(args)
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
    logger = Logger(log_file)
    logger.reset_logs()

    with gr.Blocks(title="Train 👨‍💻", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Train 👨‍💻")

        # Browse button
        masks = gr.Textbox(label="Selected Masks File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_file, outputs=masks, show_progress="hidden")

        color_map = gr.Textbox(label="Selected Color Map File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_file, outputs=color_map, show_progress="hidden")

        pre_trained_path = gr.Textbox(label="Selected Pre-trained Encoder File (Optional)")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_files, outputs=pre_trained_path, show_progress="hidden")

        with gr.Row():
            encoder_name = gr.Dropdown(label="Encoder", multiselect=False, allow_custom_value=False,
                                       choices=get_segmentation_encoders())

            freeze_encoder = gr.Slider(0.0, label="Freeze Encoder", minimum=0.0, maximum=1.0, step=0.01)

            decoder_name = gr.Dropdown(label="Decoder", multiselect=False, allow_custom_value=False,
                                       choices=get_segmentation_decoders())

        with gr.Row():
            optimizer = gr.Dropdown(label="Optimizer", multiselect=False, allow_custom_value=False,
                                    choices=get_segmentation_optimizers())

            learning_rate = gr.Slider(0.0001, label="Initial Learning Rate",
                                      minimum=0.00001, maximum=1, step=0.0001)

        with gr.Row():
            metrics = gr.Dropdown(label="Metrics", multiselect=True, allow_custom_value=False,
                                  choices=get_segmentation_metrics())

            loss_function = gr.Dropdown(label="Loss Function", multiselect=False, allow_custom_value=False,
                                        choices=get_segmentation_losses())

            augment_data = gr.Dropdown(label="Augment Data", multiselect=False, allow_custom_value=False,
                                       choices=[True, False])

        with gr.Row():
            num_epochs = gr.Number(25, label="Number of Epochs", precision=0)

            batch_size = gr.Number(8, label="Batch Size (Power of 2 Recommended)", precision=0)

            tensorboard = gr.Dropdown(label="Tensorboard", multiselect=False, allow_custom_value=False,
                                      choices=[True, False])

        # Browse button
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [masks,
                                    color_map,
                                    pre_trained_path,
                                    encoder_name,
                                    decoder_name,
                                    metrics,
                                    loss_function,
                                    freeze_encoder,
                                    optimizer,
                                    learning_rate,
                                    augment_data,
                                    num_epochs,
                                    batch_size,
                                    tensorboard,
                                    output_dir])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(exit_interface)

        with gr.Accordion("Console Logs"):
            # Add logs
            logs = gr.Code(label="", language="shell", interactive=False, container=True, lines=30)
            interface.load(logger.read_logs, None, logs, every=1)

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
    Logger(log_file).reset_logs()