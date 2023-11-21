import gradio as gr
import os
import sys
import argparse
import traceback
from tkinter import Tk, filedialog

from Tools.Common import LOG_PATH

from Tools.Download import download
from Tools.Download import get_updated_labelset_list


# ----------------------------------------------------------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------------------------------------------------------

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def read_logs():
    """

    """
    sys.stdout.flush()
    with open(LOG_PATH, "r") as f:
        return f.read()


def reset_logs():
    """

    """
    # Clear it
    with open(LOG_PATH, 'w') as file:
        pass


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def choose_directory():
    """

    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    filename = filedialog.askdirectory()
    if filename:
        if os.path.isdir(filename):
            root.destroy()
            return str(filename)
        else:
            root.destroy()
            return str(filename)
    else:
        filename = "Folder not selected"
        root.destroy()
        return str(filename)


def module_callback(username, password, source_ids, source_df, labelset_df, sources_with, output_dir, headless):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    # Convert source_ids to a list of strings
    source_ids_list = [str(id.strip()) for id in source_ids.split(",") if id.strip()]

    args = argparse.Namespace(
        username=username,
        password=password,
        source_ids=source_ids_list,
        source_df=source_df,
        labelset_df=labelset_df,
        sources_with=sources_with,
        output_dir=output_dir,
        headless=headless
    )

    try:
        # Call the download function from your module
        download(args)
        print("Done.")
    except Exception as e:
        return f"ERROR: {e}\n{traceback.format_exc()}"


# ----------------------------------------------------------------------------------------------------------------------
# Gradio
# ----------------------------------------------------------------------------------------------------------------------

demo = gr.Blocks()

with demo:
    # Title
    gr.Markdown("# CoralNet Downloader")

    # Input Parameters
    with gr.Tab("Download Source Data"):
        with gr.Row():
            username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
            password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')

        with gr.Row():
            source_ids = gr.Textbox("4085", label="Source IDs (comma-separated)")
            headless = gr.Checkbox(label="Run Browser in Headless Mode", value=True)

    with gr.Tab("Download CoralNet Dataframes"):
        with gr.Row():
            source_df = gr.Checkbox(label="Download Source DataFrame")
            labelset_df = gr.Checkbox(label="Download Labelset DataFrame")

        sources_with = gr.Dropdown(label="Sources with Labelsets",
                                   choices=get_updated_labelset_list(),
                                   multiselect=True)

    # Browse buttons
    output_dir_path = gr.Textbox("", label="Selected Output Directory")
    output_dir_button = gr.Button("Browse Output Directory")
    output_dir_button.click(choose_directory, outputs=output_dir_path, show_progress="hidden")

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
                                output_dir_path,
                                headless])

        stop = gr.Button(value="Stop")
        stop.click(fn=None, inputs=None, outputs=None, cancels=[run])

    with gr.Accordion("Console Logs"):
        logs = gr.Textbox(label="")
        demo.load(read_logs, None, logs, every=1)

if __name__ == "__main__":
    reset_logs()
    demo.launch(share=False)