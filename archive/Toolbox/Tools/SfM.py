import os
import time
import argparse
import traceback

import numpy as np

import Metashape

from Common import console_user
from Common import get_now
from Common import print_progress

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version,
                                                                      compatible_major_version))


# -----------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------

def find_files(folder, types):
    """
    Takes in a folder and a list of file types, returns a list of file paths
    that end with any of the specified extensions.
    """
    return [entry.path for entry in os.scandir(folder) if
            (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]


def sfm_workflow(args):
    """
    Takes in an input folder, runs SfM Workflow on all images in it,
    outputs the results in the output folder.
    """

    print("\n###############################################")
    print("Structure from Motion")
    print("###############################################\n")

    # Start the timer
    t0 = time.time()

    # If user passes a previous project dir use it
    if args.project_file:
        if os.path.exists(args.project_file):
            project_file = args.project_file
            project_dir = f"{os.path.dirname(project_file)}/"
        else:
            raise Exception

    elif os.path.exists(args.output_dir):
        output_dir = f"{args.output_dir}/sfm/"
        project_dir = f"{output_dir}{get_now()}/"
        os.makedirs(project_dir, exist_ok=True)
        project_file = f"{project_dir}project.psx"
    else:
        raise Exception(f"ERROR: Must provide either existing project file or output directory")

    # Create filenames for data outputs
    output_dem = project_dir + "DEM.tif"
    output_mesh = project_dir + "Mesh.ply"
    output_dense = project_dir + "Dense_Cloud.ply"
    output_ortho = project_dir + "Orthomosaic.tif"
    output_cameras = project_dir + "Cameras.xml"
    output_report = project_dir + "Report.pdf"

    # Quality checking
    if args.quality.lower() not in ["lowest", "low", "medium", "high", "highest"]:
        raise Exception(f"ERROR: Quality must be low, medium, or high")

    # ------------------------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------------------------

    # Create a metashape doc object
    doc = Metashape.Document()

    if not os.path.exists(project_file):
        print(f"NOTE: Creating new project file")
        # Create a new Metashape document and save it as a project file in the output folder.
        doc.save(project_file)
    else:
        print(f"NOTE: Opening existing project file")
        # Open existing project file.
        doc.open(project_file,
                 read_only=False,
                 ignore_lock=True,
                 archive=True)

    # Create a new chunk (3D model) in the Metashape document.
    if doc.chunk is None:
        doc.addChunk()
        doc.save()

    # Assign the chunk
    chunk = doc.chunk

    # Add the photos to the chunk
    if not chunk.cameras:

        # Check that input folder exists
        if os.path.exists(args.input_dir):
            input_dir = args.input_dir
        else:
            raise Exception("ERROR: Input directory provided doesn't exist; please check input")

        # Call the "find_files" function to get a list of photo file paths
        # with specified extensions from the image folder.
        photos = find_files(input_dir, [".jpg", ".jpeg", ".tiff", ".tif", ".png"])

        if not photos:
            raise Exception(f"ERROR: Image directory provided does not contain any usable images; please check input")

        print("\n###############################################")
        print("Adding photos")
        print("###############################################\n")

        chunk.addPhotos(photos, progress=print_progress)
        print(str(len(chunk.cameras)) + " images loaded")

        print("")
        print("Process Successful!")
        doc.save()

    # Match the photos by finding common features and establishing correspondences.
    if not chunk.tie_points:
        print("\n###############################################")
        print("Matching photos")
        print("###############################################\n")

        # Quality
        downscale = {"lowest": 8,
                     "low": 4,
                     "medium": 2,
                     "high": 1,
                     "highest": 0}[args.quality.lower()]

        chunk.matchPhotos(keypoint_limit=40000,
                          tiepoint_limit=10000,
                          generic_preselection=True,
                          reference_preselection=True,
                          downscale=downscale,
                          progress=print_progress)

        # Align the cameras to estimate their relative positions in space.
        chunk.alignCameras()

        print("")
        print("Process Successful!")
        doc.save()

    # Perform gradual selection to remove messy points
    if chunk.tie_points:
        print("\n###############################################")
        print("Performing gradual selection and camera optimization")
        print("###############################################\n")

        # Target percentage for gradual selection
        if 0 <= args.target_percentage <= 99:
            target_percentage = args.target_percentage
        else:
            raise Exception(f"ERROR: Target Percentage provided not in range [0, 99]; check input provided")

        # Obtain the tie points from the chunk
        points = chunk.tie_points.points

        # Filter selection methods
        selections = [Metashape.TiePoints.Filter.ReprojectionError,
                      Metashape.TiePoints.Filter.ReconstructionUncertainty,
                      Metashape.TiePoints.Filter.ProjectionAccuracy,
                      Metashape.TiePoints.Filter.ImageCount]

        # Loop through each of the selections, identify target percentage, remove, optimize
        for s_idx, selection in enumerate(selections):

            try:
                # Tie point filter
                f = Metashape.TiePoints.Filter()

                if s_idx == 3:
                    # ImageCount selection method
                    f.init(chunk, selection)
                    f.removePoints(1)
                else:
                    # Other selection methods
                    list_values = f.values
                    list_values_valid = list()
                    for i in range(len(list_values)):
                        if points[i].valid:
                            list_values_valid.append(list_values[i])
                    list_values_valid.sort()
                    # Find point values based on threshold
                    target = int(len(list_values_valid) * target_percentage / 100)
                    threshold = list_values_valid[target]
                    # Select and remove
                    f.selectPoints(threshold)
                    f.removePoints(threshold)

                # Optimize cameras
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True, fit_k1=True,
                                      fit_k2=True, fit_k3=True, fit_k4=True, fit_p1=True, fit_p2=True, fit_p3=True,
                                      fit_p4=True, adaptive_fitting=False, tiepoint_covariance=False)

            except Exception as e:
                print(f"WARNING: Could not filter points based on selection method {s_idx}")

        print("")
        print("Process Successful!")
        doc.save()

    # Export Camera positions
    if chunk.tie_points:
        print("\n###############################################")
        print("Exporting Camera Positions")
        print("###############################################\n")

        chunk.exportCameras(path=output_cameras,
                            progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build depth maps (2.5D representations of the scene) from the aligned photos.
    if chunk.tie_points and not chunk.depth_maps:
        print("\n###############################################")
        print("Building depth maps")
        print("###############################################\n")

        # Quality
        downscale = {"lowest": 16,
                     "low": 8,
                     "medium": 4,
                     "high": 2,
                     "highest": 1}[args.quality.lower()]

        chunk.buildDepthMaps(filter_mode=Metashape.MildFiltering,
                             downscale=downscale,
                             progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build a dense point cloud using the depth maps.
    if chunk.depth_maps and not chunk.point_cloud:
        print("\n###############################################")
        print("Building dense point cloud")
        print("###############################################\n")

        chunk.buildPointCloud(source_data=Metashape.DepthMapsData,
                              progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build a 3D model from the depth maps.
    if chunk.depth_maps and not chunk.model:
        print("\n###############################################")
        print("Building mesh")
        print("###############################################\n")

        # Quality
        facecount = {"lowest": Metashape.FaceCount.LowFaceCount,
                     "low": Metashape.FaceCount.LowFaceCount,
                     "medium": Metashape.FaceCount.MediumFaceCount,
                     "high": Metashape.FaceCount.HighFaceCount,
                     "highest": Metashape.FaceCount.HighFaceCount}[args.quality.lower()]

        chunk.buildModel(source_data=Metashape.DepthMapsData,
                         interpolation=Metashape.Interpolation.DisabledInterpolation,
                         face_count=facecount,
                         progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build a DEM from the 3D model.
    if chunk.model and not chunk.elevation:
        print("\n###############################################")
        print("Building DEM")
        print("###############################################\n")

        chunk.buildDem(source_data=Metashape.ModelData,
                       interpolation=Metashape.Interpolation.DisabledInterpolation,
                       progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build an orthomosaic from the 3D model.
    if chunk.model and not chunk.orthomosaic:
        print("\n###############################################")
        print("Building orthomosaic")
        print("###############################################\n")

        # Create the orthomosaic
        chunk.buildOrthomosaic(surface_data=Metashape.ModelData,
                               blending_mode=Metashape.BlendingMode.MosaicBlending,
                               fill_holes=False,
                               progress=print_progress)

        print("")
        print("Process Successful!")
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
                               crs=chunk.crs,
                               progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Export the mesh if it exists in the chunk.
    if chunk.model and not os.path.exists(output_mesh):
        print("\n###############################################")
        print("Exporting mesh")
        print("###############################################\n")

        chunk.exportModel(path=output_mesh, progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Export the DEM if it exists in the chunk.
    if chunk.elevation and not os.path.exists(output_dem):
        print("\n###############################################")
        print("Exporting DEM")
        print("###############################################\n")

        chunk.exportRaster(path=output_dem,
                           source_data=Metashape.ElevationData,
                           progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Export the orthomosaic as a TIFF file if it exists in the chunk.
    if chunk.orthomosaic and not os.path.exists(output_ortho):
        print("\n###############################################")
        print("Exporting orthomosaic")
        print("###############################################\n")

        # Set compression parameters (otherwise bigtiff error)
        compression = Metashape.ImageCompression()
        compression.tiff_big = True

        chunk.exportRaster(path=output_ortho,
                           source_data=Metashape.OrthomosaicData,
                           image_compression=compression,
                           progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Finally, export the report
    print("\n###############################################")
    print("Exporting Report")
    print("###############################################\n")
    chunk.exportReport(path=output_report)

    print("")
    print("Process Successful!")
    doc.save()

    # Print a message indicating that the processing has finished and the results have been saved.
    print(f"\nNOTE: Processing finished, results saved to {project_dir}")
    print(f"NOTE: Completed in {np.around(((time.time() - t0) / 60), 2)} minutes")


def sfm(args):
    """

    """
    import Metashape

    # Get the License from the user
    if args.metashape_license is not None:
        metashape_license = args.metashape_license
    else:
        metashape_license = os.getenv('METASHAPE_LICENSE')

    if metashape_license in ["", None]:
        raise Exception("ERROR: You must pass in a Metashape License.")

    try:
        # Get the Metashape License stored in the environmental variable
        Metashape.License().activate(metashape_license)
        # Run the workflow
        sfm_workflow(args)

    except Exception as e:
        print(f"{e}\nERROR: Could not finish workflow!")
        print(traceback.print_exc())

    finally:
        # Deactivate the license
        Metashape.License().deactivate()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    """

    """

    parser = argparse.ArgumentParser(description='SfM Workflow')

    parser.add_argument('--metashape_license', type=str,
                        default=os.getenv('METASHAPE_LICENSE'),
                        help='The Metashape License.')

    parser.add_argument('--input_dir', type=str,
                        help='Path to the input folder containing images.')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output folder.')

    parser.add_argument('--project_file', type=str,
                        help='Path to existing Metashape project file (.psx).')

    parser.add_argument('--quality', type=str, default="Medium",
                        help='Quality of data products [Lowest, Low, Medium, High, Highest]')

    parser.add_argument('--target_percentage', type=int, default=75,
                        help='Percentage of points to target for each gradual selection method')

    args = parser.parse_args()

    try:
        # Run the workflow
        sfm(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()