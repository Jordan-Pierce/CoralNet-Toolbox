import os
import sys
import time

import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Toolbox.Tools import *

import Metashape

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


# -----------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------

def find_files(folder, types):
    """Takes in a folder and a list of file types, returns a list of file paths
    that end with any of the specified extensions."""
    return [entry.path for entry in os.scandir(folder) if
            (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]


def run_workflow(args):
    """Takes in an input folder, runs SfM Workflow on all images in it,
    outputs the results in the output folder."""

    print("\n###############################################")
    print("Structure from Motion")
    print("###############################################\n")

    # Start the timer
    global t0
    t0 = time.time()

    # Check that input folder exists
    if os.path.exists(args.input_dir):
        input_dir = args.input_dir
    else:
        print("ERROR: Input directory provided doesn't exist; please check input")
        sys.exit(1)

    # Create the output folder if it doesn't already exist
    output_dir = f"{args.output_dir}\\sfm\\{get_now()}\\"
    os.makedirs(output_dir, exist_ok=True)
    # Create filenames for data outputs
    output_dem = output_dir + "/DEM.tif"
    output_mesh = output_dir + "/Mesh.ply"
    output_dense = output_dir + "/Dense_Cloud.ply"
    output_orthomosaic = output_dir + "/Orthomosaic.tif"

    # Call the "find_files" function to get a list of photo file paths
    # with specified extensions from the image folder.
    photos = find_files(input_dir, [".jpg", ".jpeg", ".tiff", ".tif", ".png"])

    # Create a metashape doc object
    doc = Metashape.Document()

    if not os.path.exists(output_dir + "/project.psx"):
        # Create a new Metashape document and save it as a project file in the output folder.
        doc.save(output_dir + '/project.psx')
    else:
        # Else open the existing one
        doc.open(output_dir + '/project.psx',
                 read_only=False,
                 ignore_lock=True,
                 archive=True)

    # Create a new chunk (3D model) in the Metashape document.
    if doc.chunk is None:
        doc.addChunk()
        doc.save()

    # Assign the chunk
    chunk = doc.chunk

    # Add the photos to the chunk.
    if not chunk.cameras:

        print("\n###############################################")
        print("Adding photos")
        print("###############################################\n")

        chunk.addPhotos(photos)
        print(str(len(chunk.cameras)) + " images loaded")
        doc.save()

    # Match the photos by finding common features and establishing correspondences.
    if not chunk.tie_points:

        print("\n###############################################")
        print("Matching photos")
        print("###############################################\n")

        chunk.matchPhotos(keypoint_limit=40000,
                          tiepoint_limit=10000,
                          generic_preselection=True,
                          reference_preselection=True,
                          downscale=4)

        # Align the cameras to estimate their relative positions in space.
        chunk.alignCameras()
        doc.save()

    # Build depth maps (2.5D representations of the scene) from the aligned photos.
    if chunk.tie_points and not chunk.depth_maps:

        print("\n###############################################")
        print("Building depth maps")
        print("###############################################\n")

        chunk.buildDepthMaps(filter_mode=Metashape.MildFiltering,
                             downscale=1)
        doc.save()

    # Build a dense point cloud using the depth maps.
    if chunk.depth_maps and not chunk.point_cloud:

        print("\n###############################################")
        print("Building dense point cloud")
        print("###############################################\n")

        chunk.buildPointCloud(source_data=Metashape.DepthMapsData)
        doc.save()

    # Build a 3D model from the depth maps.
    if chunk.depth_maps and not chunk.model:

        print("\n###############################################")
        print("Building mesh")
        print("###############################################\n")

        chunk.buildModel(source_data=Metashape.DepthMapsData,
                         face_count=Metashape.FaceCount.HighFaceCount)
        doc.save()

    # Build a DEM from the 3D model.
    if chunk.model and not chunk.elevation:

        print("\n###############################################")
        print("Building DEM")
        print("###############################################\n")

        chunk.buildDem(source_data=Metashape.ModelData)
        doc.save()

    # Build an orthomosaic from the 3D model.
    if chunk.model and not chunk.orthomosaic:

        print("\n###############################################")
        print("Building orthomosaic")
        print("###############################################\n")

        # Create the orthomosaic
        chunk.buildOrthomosaic(surface_data=Metashape.ModelData,
                               blending_mode=Metashape.BlendingMode.MosaicBlending)
        # Save the document
        doc.save()

    # Export the dense point cloud if it exists in the chunk.
    if chunk.point_cloud and not os.path.exists(output_dense):

        print("\n###############################################")
        print("Exporting dense point cloud")
        print("###############################################\n")

        chunk.exportPointCloud(path=output_dense,
                               save_point_color=True,
                               save_point_classification=True,
                               save_point_normal=True,
                               save_point_confidence=True,
                               crs=chunk.crs)

    # Export the mesh if it exists in the chunk.
    if chunk.model and not os.path.exists(output_mesh):

        print("\n###############################################")
        print("Exporting mesh")
        print("###############################################\n")

        chunk.exportModel(path=output_mesh,)

    # Export the DEM if it exists in the chunk.
    if chunk.elevation and not os.path.exists(output_dem):

        print("\n###############################################")
        print("Exporting DEM")
        print("###############################################\n")

        chunk.exportRaster(path=output_dem,
                           source_data=Metashape.ElevationData)

    # Export the orthomosaic as a GeoTIFF file if it exists in the chunk.
    if chunk.orthomosaic and not os.path.exists(output_orthomosaic):

        print("\n###############################################")
        print("Exporting orthomosaic")
        print("###############################################\n")

        chunk.exportRaster(path=output_orthomosaic,
                           source_data=Metashape.OrthomosaicData)

    # Print a message indicating that the processing has finished and the results have been saved.
    print(f"NOTE: Processing finished, results saved to {output_dir}")
    print(f"NOTE: Completed in {((time.time() - t0) / 60)}")


def sfm(args):
    """

    """
    # Get the License from the user
    if args.metashape_license is not None:
        metashape_license = args.metashape_license
    else:
        metashape_license = os.getenv('METASHAPE_LICENSE')

    if metashape_license in ["", None]:
        raise Exception("ERROR: You must pass in a Metashape License.")

    # Get the Metashape License stored in the environmental variable
    Metashape.License().activate(metashape_license)

    try:
        # Run the workflow
        run_workflow(args)
        print("Done.")

    except Exception as e:
        print(f"{e}\nERROR: Could not finish workflow!")

    finally:
        # Always deactivate after script
        print("NOTE: Deactivating License...")
        Metashape.License().deactivate()

        if not Metashape.License().valid:
            print("NOTE: License Deactivated or was not Active to begin with.")
        else:
            print("ERROR: License was not Deactivated!")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='SfM Workflow')

    parser.add_argument('--input_dir', type=str,
                        help='Path to the input folder containing images.')

    parser.add_argument('--output_dir', type=str,
                        help='Path to the output folder.')

    parser.add_argument('--metashape_license', type=str,
                        help='The Metashape License.')

    args = parser.parse_args()

    try:
        # Run the workflow
        sfm(args)
        print("Done.")

    except Exception as e:
        print(f"{e}\nERROR: Could not finish workflow!")


if __name__ == '__main__':
    main()


