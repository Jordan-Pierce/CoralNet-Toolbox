import gradio as gr

import os
import sys
import time
import argparse
import traceback

from Toolbox.Pages.common import js
from Toolbox.Pages.common import Logger
from Toolbox.Pages.common import read_logs
from Toolbox.Pages.common import reset_logs
from Toolbox.Pages.common import choose_files
from Toolbox.Pages.common import choose_directory
from Toolbox.Pages.common import get_port

from Toolbox.Tools.Common import DATA_DIR
from Toolbox.Tools.Common import LOG_PATH

from Toolbox.Tools.Classification import classification
from Toolbox.Tools.Classification import get_classifier_losses
from Toolbox.Tools.Classification import get_classifier_encoders

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(patches, output_dir, encoder_name, loss_function, weighted_loss, augment_data, dropout_rate, num_epochs,
                    batch_size, learning_rate, tensorboard):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        patches=patches.split(" "),
        output_dir=output_dir,
        encoder_name=encoder_name,
        loss_function=loss_function,
        weighted_loss=weighted_loss,
        augment_data=augment_data,
        dropout_rate=dropout_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tensorboard=tensorboard,
    )

    try:
        # Call the function
        classification(args)
        print("Done.")
    except Exception as e:
        print(f"ERROR: {e}\n{traceback.format_exc()}")


# ----------------------------------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------------------------------
def check_interface():
    """

    """
    global RESTART
    RESTART = True

    return


def exit_interface():
    """

    """
    reset_logs()

    print("")
    print("Stopped program successfully!")
    print("Connection closed!")
    print("")
    print("Please close the browser tab.")
    time.sleep(1)
    sys.exit(1)


def create_interface():
    """

    """
    reset_logs()

    with gr.Blocks(title="Train 👨‍💻", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("Train 👨‍💻")

        with gr.Group():

            patches = gr.Textbox(label="Selected Patch Files")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_files, outputs=patches, show_progress="hidden")

            # Browse button
            output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Group():
            with gr.Row():

                encoder_name = gr.Dropdown(label="Encoder", multiselect=False,
                                           choices=get_classifier_encoders())

                loss_function = gr.Dropdown(label="Loss Function", multiselect=False,
                                            choices=get_classifier_losses())

                weighted_loss = gr.Checkbox(label="Weighted Loss", value=True)

            with gr.Row():

                augment_data = gr.Checkbox(label="Augment Data", value=False)

                dropout_rate = gr.Slider(0, label="Dropout Rate", interactive=True,
                                         minimum=0, maximum=1, step=0.1)

            with gr.Row():

                num_epochs = gr.Number(25, label="Number of Epochs", precision=0)

                batch_size = gr.Number(128, label="Batch Size", precision=0)

                learning_rate = gr.Number(0.0005, label="Initial Learning Rate")

            tensorboard = gr.Checkbox(label="Tensorboard", value=True)

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [patches,
                                    output_dir,
                                    encoder_name,
                                    loss_function,
                                    weighted_loss,
                                    augment_data,
                                    dropout_rate,
                                    num_epochs,
                                    batch_size,
                                    learning_rate,
                                    tensorboard])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(check_interface)

        with gr.Accordion("Console Logs"):
            logs = gr.Code(label="", language="shell", interactive=False, container=True)
            interface.load(read_logs, None, logs, every=1)

    interface.launch(prevent_thread_lock=True, server_port=get_port(), inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

interface = create_interface()

while True:
    time.sleep(0.5)
    if RESTART:
        exit_interface()