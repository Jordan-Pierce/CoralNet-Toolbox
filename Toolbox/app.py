import gradio as gr
import os
import sys
import argparse
import traceback
from tkinter import Tk, filedialog

from Tools.Common import LOG_PATH

from Tools.Download import download


# ----------------------------------------------------------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------------------------------------------------------

class Logger:
    def __init__(self, filename):
        # Clear it
        with open(filename, 'w') as file:
            pass

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


def main_interface(username, password, source_ids, source_df, labelset_df, sources_with, output_dir, headless):
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


def download_callback(username, password, source_ids, source_df, labelset_df, sources_with, output_dir, headless):
    main_interface(username, password, source_ids, source_df, labelset_df, sources_with, output_dir, headless)


# ----------------------------------------------------------------------------------------------------------------------
# Gradio
# ----------------------------------------------------------------------------------------------------------------------

demo = gr.Blocks()

with demo:
    # Title
    gr.Markdown("## CoralNet Downloader")

    # Input Parameters
    username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
    password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')
    source_ids = gr.Textbox("4085", label="Source IDs (comma-separated)")
    source_df = gr.Checkbox(label="Download Source DataFrame")
    labelset_df = gr.Checkbox(label="Download Labelset DataFrame")
    sources_with = gr.Textbox(label="Sources with Labelsets (comma-separated)", placeholder="e.g., Labelset1,Labelset2")
    headless = gr.Checkbox(label="Run Browser in Headless Mode")

    # Browse buttons
    output_dir_button = gr.Button("Browse Output Directory")
    output_dir_path = gr.Textbox("", label="Selected Output Directory")
    output_dir_button.click(choose_directory, outputs=output_dir_path, show_progress="hidden")

    # Run button (callback)
    run_button = gr.Button("Run")
    run_button.click(download_callback,
                     [username,
                      password,
                      source_ids,
                      source_df,
                      labelset_df,
                      sources_with,
                      output_dir_path,
                      headless])

    logs = gr.Textbox()
    demo.load(read_logs, None, logs, every=1)

if __name__ == "__main__":
    demo.launch(share=False)