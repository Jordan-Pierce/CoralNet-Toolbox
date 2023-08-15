import os
import sys
import time
import json
import argparse

from plyfile import PlyData, PlyElement

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
def find_closest_color(color, color_map):
    """

    """
    # Get the distance from color to all possible colors
    distances = np.linalg.norm(color_map - color, axis=1)

    # Find the index of the color with the minimum distance
    closest_color_index = np.argmin(distances)

    # Get the closest color by index
    closest_color = color_map[closest_color_index]

    return closest_color


def post_process_pcd(temp_path, dense_path, color_map):
    """

    """
    # Open the temp pcd file, and write to the post-processed file
    with open(temp_path, 'rb') as f_in, open(dense_path, 'wb') as f_out:

        # Open the header
        plydata = PlyData.read(f_in)
        # Get the vertices, num points
        vertex_data = plydata['vertex']
        num_points = vertex_data['x'].shape[0]
        # Array to hold the updated colors
        updated_colors = np.empty((num_points, 3), dtype=np.uint8)

        for i in range(num_points):

            # Get the color values
            red = vertex_data['red'][i]
            green = vertex_data['green'][i]
            blue = vertex_data['blue'][i]
            color = np.array([red, green, blue])
            # Modify based on distance to actual color
            updated_colors[i] = find_closest_color(color, color_map)

            # Gooey
            print_progress(i, len(num_points))

        # Add new colors to vertex arrays
        vertex_data['red'] = updated_colors[:, 0]
        vertex_data['green'] = updated_colors[:, 1]
        vertex_data['blue'] = updated_colors[:, 2]

        # Save to dense path
        plydata_out = PlyData([PlyElement.describe(vertex_data, 'vertex')], text=False)
        plydata_out.write(f_out)

    if os.path.exists(dense_path):
        print("NOTE: Post-processed point cloud saved successfully")
    else:
        raise Exception("ERROR: Issue with saving post-processed point cloud")


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

        # TODO Change to delete in the future
        if classified_chunk is None:

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
        # Remove the temp file
        if os.path.exists(output_dense):
            os.remove(output_temp)
        else:
            raise Exception("ERROR: Classified dense point cloud not found")

        print("\n###############################################")
        print("Importing post-processed classified point cloud")
        print("###############################################\n")
        # Import the updated version
        classified_chunk.importPointCloud(output_dense,
                                          replace_asset=True)

    except Exception as e:
        print(f"ERROR: Could not complete post-processing of classified point cloud\n{e}")
        sys.exit(1)

    try:
        print("\n###############################################")
        print("Classifying mesh")
        print("###############################################\n")
        # Classify (colorize) the mesh using the classified dense point cloud.
        # Update the mesh to apply the new colorization settings
        classified_chunk.colorizeModel(Metashape.PointCloudData)
        doc.save()

    except Exception as e:
        print(f"ERROR: Could not classify mesh\n{e}")
        sys.exit(1)

    # Export the mesh if it exists in the chunk.
    if classified_chunk.model and not os.path.exists(output_mesh):
        print("\n###############################################")
        print("Exporting classified mesh")
        print("###############################################\n")

        classified_chunk.exportModel(path=output_mesh, progress=print_sfm_progress)

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

    parser.add_argument('--project_file', type=str,
                        help='Path to the existing Metashape project file (.psx)')

    parser.add_argument('--masks_dir', type=str,
                        help='Directory containing color masks for images.')

    parser.add_argument('--color_map', type=str,
                        help='Path to Color Map JSON file.')

    parser.add_argument('--chunk_index', type=int, default=0,
                        help='Index of chunk to classify (0-based indexing)')

    parser.add_argument('--metashape_license', type=str,
                        default=os.getenv('METASHAPE_LICENSE'),
                        help='The Metashape License.')

    args = parser.parse_args()

    try:
        # Run the workflow
        seg3d(args)
        print("Done.")

    except Exception as e:
        print(f"{e}\nERROR: Could not finish workflow!")


if __name__ == '__main__':
    main()
