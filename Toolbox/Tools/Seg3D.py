import os
import sys
import time
import json
import argparse

from plyfile import PlyData, PlyElement
from scipy.spatial.distance import cdist

from Toolbox.Tools import *
from Toolbox.Tools.SfM import print_sfm_progress

import Metashape

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


# -----------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------

def find_closest_color(color_array, color_map):
    """

    """
    # Distances for each color in array, to each color in color_map
    distances = cdist(color_array, color_map)
    # Index for closest color in color_map for each color in array
    closest_color_indices = np.argmin(distances, axis=1)
    # Closest colors
    closest_colors = color_map[closest_color_indices]

    return closest_colors


def post_process_pcd(temp_path, dense_path, color_map, chunk_size=10000000):
    """

    """
    # Get the header of file
    plydata = PlyData.read(temp_path)
    # Get the vertices
    vertex_data = plydata['vertex']
    # total number of points
    num_points = vertex_data['x'].shape[0]
    # For memory, in batches, get updated color values
    for i in range(0, num_points, chunk_size):
        # last index in batch
        chunk_end = min(i + chunk_size, num_points)
        # colors of batch
        red_chunk = vertex_data['red'][i:chunk_end]
        green_chunk = vertex_data['green'][i:chunk_end]
        blue_chunk = vertex_data['blue'][i:chunk_end]
        # Stacking and getting the closest values
        color_array = np.column_stack((red_chunk, green_chunk, blue_chunk))
        modified_colors = find_closest_color(color_array, color_map)
        # Updating vertex colors
        vertex_data['red'][i:chunk_end] = modified_colors[:, 0]
        vertex_data['green'][i:chunk_end] = modified_colors[:, 1]
        vertex_data['blue'][i:chunk_end] = modified_colors[:, 2]
        # Gooey
        print_progress(i, num_points)

    print("NOTE: Writing post-processed point cloud to disk")
    plydata.write(dense_path)

    if os.path.exists(dense_path):
        print("NOTE: Post-processed point cloud saved successfully")
    else:
        raise Exception("ERROR: Issue with saving post-processed point cloud")

    # Close the temp file
    plydata = None


def seg3d_workflow(args):
    """

    """
    print("\n###############################################")
    print("3D Semantic Segmentation")
    print("###############################################\n")

    # Start the timer
    t0 = time.time()

    # Existing project file
    if os.path.exists(args.project_file):
        project_file = args.project_file
        project_dir = f"{os.path.dirname(project_file)}\\"
    else:
        print(f"ERROR: Project file provided doesn't exist; check input provided")
        sys.exit(1)

    # Segmentation masks for images in project
    if os.path.exists(args.masks_dir):
        masks_dir = f"{args.masks_dir}\\"
    else:
        print(f"ERROR: Masks directory provided doesn't exist; check input provided")
        sys.exit(1)

    # Color mapping file
    if os.path.exists(args.color_map):
        with open(args.color_map, 'r') as json_file:
            color_map = json.load(json_file)

        # Modify color map format
        color_map = np.array([v['color'] for k, v in color_map.items()])

    else:
        print(f"ERROR: Color Mapping JSON file provided doesn't exist; check input provided")
        sys.exit(1)

    # Create filenames for data outputs
    output_temp = project_dir + "Temp_Cloud_Classified.ply"
    output_dense = project_dir + "Dense_Cloud_Classified.ply"
    output_mesh = project_dir + "Mesh_Classified.ply"
    output_ortho = project_dir + "Orthomosaic_Classified.tif"

    # ------------------------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------------------------

    # Create a metashape doc object
    doc = Metashape.Document()

    print(f"NOTE: Opening existing project file")
    # Open existing project file
    doc.open(project_file,
             read_only=False,
             ignore_lock=True,
             archive=True)

    try:
        # Get the chunk
        chunk = doc.chunks[args.chunk_index]

        if not chunk.point_cloud:
            print(f"ERROR: Chunk does not contain a dense point cloud; exiting")
            sys.exit(1)

        # Loop through the chunks to see if there is already
        # a classified chunk for this chunk index, else duplicate
        classified_chunk = None

        for c in doc.chunks:
            if c.label == chunk.label + " Classified":
                classified_chunk = c
                break

        # If the classified chunk already exists, that would be an issue
        # Make the user decide if they want to delete it, or change name
        if classified_chunk:
            print("WARNING: Classified chunk for index provided already exists.")
            print("WARNING: Delete or change the name of this chunk before running Seg3D.")
            sys.exit(1)

        print("\n###############################################")
        print("Duplicating chunk")
        print("###############################################\n")
        # Create a copy to serve as the classified chunk
        classified_chunk = chunk.copy(progress=print_sfm_progress)
        # Rename classified chunk
        classified_chunk.label = chunk.label + " Classified"

        # Save doc
        doc.save()

    except Exception as e:
        print(f"ERROR: Could not create a duplicate of chunk {args.chunk_index}\n{e}")
        sys.exit()

    try:
        # Make sure there are cameras
        if not classified_chunk.cameras:
            print(f"ERROR: Duplicate of chunk does not contain any camera paths; exiting")
            sys.exit(1)

        print("\n###############################################")
        print("Updating camera paths")
        print("###############################################\n")
        # Update all the photo paths in the classified chunk to be the labels

        for camera in classified_chunk.cameras:
            # The name of segmentation mask
            classified_photo = f"{masks_dir}{os.path.basename(camera.photo.path)}"
            # Check that it exists
            if os.path.exists(classified_photo):
                camera.photo.path = classified_photo
            else:
                print(f"ERROR: Could not find the following file {classified_photo}; exiting")
                sys.exit(1)

        # Save the document
        doc.save()

    except Exception as e:
        print(f"ERROR: Could not update camera paths\n{e}")
        sys.exit()

    try:
        print("\n###############################################")
        print("Classifying dense point cloud")
        print("###############################################\n")
        # Classify (colorize) the dense point cloud using the labels.
        # Update the point cloud to apply the new colorization settings
        classified_chunk.colorizePointCloud(Metashape.ImagesData,
                                            progress=print_sfm_progress)

        doc.save()

    except Exception as e:
        print(f"ERROR: Could not classify dense point cloud\n{e}")
        sys.exit(1)

    try:
        print("\n###############################################")
        print("Exporting classified dense point cloud")
        print("###############################################\n")
        # First export the point cloud
        classified_chunk.exportPointCloud(path=output_temp,
                                          save_point_color=True,
                                          save_point_classification=True,
                                          save_point_normal=True,
                                          save_point_confidence=True,
                                          crs=classified_chunk.crs,
                                          progress=print_sfm_progress)

        print("\n###############################################")
        print("Post-processing classified point cloud")
        print("###############################################\n")
        # Edit dense point cloud colors
        post_process_pcd(output_temp, output_dense, color_map)

        try:
            # Remove the temp file
            if os.path.exists(output_dense):
                os.remove(output_temp)
        except:
            print("WARNING: Could not delete temp point cloud file; please delete it")

        print("\n###############################################")
        print("Importing post-processed classified point cloud")
        print("###############################################\n")
        # Import the updated version
        classified_chunk.importPointCloud(output_dense,
                                          replace_asset=True,
                                          progress=print_sfm_progress)

        doc.save()

    except Exception as e:
        print(f"ERROR: Could not complete post-processing of classified point cloud\n{e}")
        sys.exit(1)

    if args.classify_mesh and classified_chunk.model:
        try:
            print("\n###############################################")
            print("Classifying mesh")
            print("###############################################\n")
            # Classify (colorize) the mesh using the classified dense point cloud.
            # Update the mesh to apply the new colorization settings
            classified_chunk.colorizeModel(Metashape.PointCloudData,
                                           progress=print_sfm_progress)
            # Save the document
            doc.save()

        except Exception as e:
            print(f"ERROR: Could not classify mesh\n{e}")
            sys.exit(1)

    if args.classify_ortho and classified_chunk.elevation:
        try:
            print("\n###############################################")
            print("Classifying orthomosaic")
            print("###############################################\n")

            # Create the orthomosaic
            chunk.buildOrthomosaic(surface_data=Metashape.ElevationData,
                                   blending_mode=None,
                                   fill_holes=False,
                                   progress=print_sfm_progress)
            # Save the document
            doc.save()

        except Exception as e:
            print(f"ERROR: Could not classify orthomosiac\n{e}")
            sys.exit(1)

    # Export the mesh if it exists in the chunk.
    if args.classify_mesh and classified_chunk.model and not os.path.exists(output_mesh):
        print("\n###############################################")
        print("Exporting classified mesh")
        print("###############################################\n")

        classified_chunk.exportModel(path=output_mesh,
                                     progress=print_sfm_progress)

    if args.clasify_ortho and classified_chunk.orthomosaic and not os.path.exists(output_ortho):
        print("\n###############################################")
        print("Exporting classified orthomosaic")
        print("###############################################\n")

        # Set compression parameters (otherwise bigtiff error)
        compression = Metashape.ImageCompression()
        compression.tiff_big = True

        chunk.exportRaster(path=output_ortho,
                           source_data=Metashape.OrthomosaicData,
                           image_compression=compression,
                           progress=print_sfm_progress)

    # Print a message indicating that the processing has finished and the results have been saved.
    print(f"NOTE: Processing finished, results saved to {project_dir}")
    print(f"NOTE: Completed in {np.around(((time.time() - t0) / 60), 2)} minutes")


def seg3d(args):
    """

    """
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
        seg3d_workflow(args)

    except Exception as e:
        print(f"{e}\nERROR: Could not finish workflow!")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='Seg3D Workflow')

    parser.add_argument('--metashape_license', type=str,
                        default=os.getenv('METASHAPE_LICENSE'),
                        help='The Metashape License.')

    parser.add_argument('--project_file', type=str,
                        help='Path to the existing Metashape project file (.psx)')

    parser.add_argument('--masks_dir', type=str,
                        help='Directory containing color masks for images.')

    parser.add_argument('--color_map', type=str,
                        help='Path to Color Map JSON file.')

    parser.add_argument('--chunk_index', type=int, default=0,
                        help='Index of chunk to classify (0-based indexing)')

    parser.add_argument('--classify_mesh', action='store_true',
                        help='Classify mesh using dense point cloud')

    parser.add_argument('--classify_ortho', action='store_true',
                        help='Classify orthomosaic using mesh')

    args = parser.parse_args()

    try:
        # Run the workflow
        seg3d(args)
        print("Done.")

    except Exception as e:
        print(f"{e}\nERROR: Could not finish workflow!")


if __name__ == '__main__':
    main()
