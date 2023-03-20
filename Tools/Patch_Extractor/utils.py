import os
import pandas as pd


def txt_to_csv(path, patch_size):
    """This function takes in the path of a txt file output by the Patch
    Extractor tool, and converts it into the format required by CoralNet.
    The output is a pandas dataframe"""

    csv = pd.read_csv(path, header=None, delimiter="\t")
    csv.columns = ['File_Path', 'Column', 'Row', 'Label']

    csv['Name'] = [os.path.basename(_) for _ in csv['File_Path'].values]
    csv['Label'] = [_.split("_")[0] for _ in csv['Label'].values]
    csv['Column'] = csv['Column'].values + (patch_size // 2)
    csv['Row'] = csv['Row'].values + (patch_size // 2)

    return csv