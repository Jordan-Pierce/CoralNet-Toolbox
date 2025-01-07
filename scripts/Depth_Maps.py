# Exports depth map of each camera.
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape

import os
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportDepthMaps():

    def __init__(self, project_file, output_dir, output_name):
        self.project_file = project_file
        self.output_dir = f"{output_dir}/{output_name}"

        assert os.path.exists(self.project_file)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the project
        self.doc = Metashape.Document()
        self.doc.open(self.project_file)

        # Get the active chunk
        self.chunk = self.doc.chunk
        
        # Get the camera list
        self.camera_list = self.get_camera_list()
        # Get the depth maps
        self.depth_maps = self.get_depth_maps()
        
    def get_camera_list(self):
        return [c for c in self.chunk.cameras if (c.transform and c.type == Metashape.Camera.Type.Regular)]

    def get_depth_maps(self):
        return {c: self.chunk.depth_maps[c] for c in self.camera_list if c in self.chunk.depth_maps.keys()}
    
    def process_camera(self, camera, depth):
        # Get camera image dimensions
        height = camera.photo.image().height
        width = camera.photo.image().width
        # Resize to original camera image dimensions (nearest neighbor interpolation)
        depth = depth.resize(width, height)
        # Convert depth map to float16, compress and save
        depth = depth.convert(" ", "F16")
        compr = Metashape.ImageCompression()
        compr.tiff_compression = Metashape.ImageCompression().TiffCompressionDeflate
        depth.save(f"{self.output_dir}/{camera.label}.png", compression=compr)
        return camera.label

    def export_depth_maps(self):
        print("Script started...")
        # Create a pool of workers
        pool = mp.Pool(processes=mp.cpu_count())
        
        # Process cameras in parallel
        results = list(tqdm(
            pool.starmap(self.process_camera, 
                [(camera, self.depth_maps[camera]) for camera in self.camera_list if camera in self.depth_maps]),
            total=len(self.depth_maps)
        ))
        # Close the pool
        pool.close()
        pool.join()
        
        # Count successful exports
        count = sum(1 for r in results if r is not None)
        
        print(f"Script finished. Total cameras processed: {count}")
        print(f"Depth maps exported to:\n {self.output_dir}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description='Export depth maps from Metashape project'
    )

    parser.add_argument("--project_file", required=True, help="Path to the Metashape project file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_name", required=True, help="Output name")

    args = parser.parse_args()

    exporter = ExportDepthMaps(
        project_file=args.project_file,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
    exporter.export_depth_maps()


if __name__ == '__main__':
    main()
