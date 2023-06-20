import os
import sys
import glob
sys.path.append('../')

import pandas as pd
from thefuzz import fuzz
from thefuzz import process

# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

# Get this script's directory
VISCORE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapping from VPI Table to CoralNet
FUNC_GROUPS_MAP = {
    'Hard_coral': 'Hard coral',
    'Ascidian': 'Other Invertebrates',
    'Non_biological': 'Other',
    'Invertebrates': 'Other Invertebrates',
    'Soft_coral': 'Other Invertebrates',
    'Anemone': 'Other Invertebrates',
    'Sponge': 'Other Invertebrates',
    'Coral_skeleton': 'Hard Substrate',
    'Seagrass': 'Seagrass',
    'Macroalgae': 'Algae',
    'CCA': 'Hard Substrate',
    'Cyanobacteria': 'Algae',
    'Turf_Algae': 'Algae'
}

# Boilerplate information for the labelset description
# Additional information regarding the label should be appended
DESCRIPTION = f"A collaboration between NOAA’s Office of Habitat Conservation and Florida Keys " \
              "National Marine Sanctuary, the National Marine Sanctuary Foundation, the State of" \
              " Florida, Coral Restoration Foundation, Mote Marine Laboratory and Aquarium, The " \
              "Florida Aquarium, The Nature Conservancy, Reef Renewal and University of Florida" \
              ".\n\nTogether, the partners will restore nearly three million square feet of the " \
              "Florida Reef Tract, about the size of 52 football fields, at seven key reef sites" \
              ". It is one of the largest strategies ever proposed in the field of coral " \
              "restoration.\n\nMission: Iconic Reefs builds off of decades of pioneering " \
              "restoration efforts proven successful in the Florida Keys involving growing and " \
              "transplanting corals, setting the stage for this large-scale, multi-phased " \
              "restoration effort at seven reefs.\n\nFor more information visit: " \
              "https://www.fisheries.noaa" \
              ".gov/feature-story/mission-iconic-reefs-shares-strategic-priorities-2022-2025" \
              "\nMission Iconic Reef point annotations are created in VISCORE using the Virtual " \
              "Point Intercept (VPI) tool, which is used for estimating percent coverage " \
              "abundance in 3D space. Briefly, points (~2500) are randomly placed in a 3D " \
              "reconstructed scene (i.e., SfM point cloud) which are then annotated by trained " \
              "professionals. Points are then projected to the appropriate images, and used as " \
              "point annotations for CoralNet. For questions on methodology, please contact the " \
              "Mission Iconic Reef team at the National Center for Coastal and Ocean Sciences."

try:
    # Get the VPI Lookup table
    vpi_table = f"{VISCORE_DIR}\Caribbean_VPI_Lookup_V3_V4.csv"
    vpi_table = pd.read_csv(vpi_table, index_col=0)
    # Deal with labels that a NULL
    vpi_table.dropna(inplace=True)
    vpi_table = vpi_table[~vpi_table['VPI_name'].str.contains('n/a')]

except Exception as e:
    print("ERROR: Check that the VPI Lookup table is correct")
    print("ERROR: Place the *VPI_Lookup*.csv file in the VISCORE folder")
    sys.exit(1)



