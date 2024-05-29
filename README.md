# 3D-Surface-Reconstruction
3D Surface and Mesh Reconstruction from RGB Videos using COLMAP and Open3D

This repository contains a Python script for performing 3D reconstruction from a set of images using the COLMAP and Open3D libraries with RGB video frames. The process for reconstruction includes dense reconstruction, point cloud filtering and segmentation, surface reconstruction, mesh refinement, and visualization.

## Features

- Dense reconstruction using COLMAP's patch match stereo algorithm
- Point cloud filtering to remove statistical outliers
- Point cloud segmentation using DBSCAN clustering to separate objects
- Surface reconstruction using Poisson surface reconstruction
- Mesh refinement using Laplacian and Taubin smoothing
- Visualization of the reconstructed objects and the final merged mesh

## Dependencies

- Python 3.x
- COLMAP
- pycolmap
- Open3D
- NumPy

## Usage

1. Install the required dependencies:
   ```
   pip install pycolmap open3d numpy
   ```

2. Prepare your input data:
   - Place the input color images in the `dataset_path` directory specified in the script.
   - Update the `dataset_path` variable in the script to point to your dataset directory.

3. Run the script:
   ```python
   python dense_reconstruction.py
   ```

4. The script will perform the following steps:
   - Sparse reconstruction using the color images (if not already done)
   - Dense reconstruction using the sparse reconstruction data and color images
   - Filtering and segmentation of the dense point cloud
   - Surface reconstruction and refinement for each segmented object
   - Visualization of the reconstructed objects
   - Merging of the object meshes into a single mesh
   - Saving the final mesh as `dense_reconstruction_mesh.ply`

## Pipeline Steps

1. **Sparse Reconstruction:**
- The sparse reconstruction step takes the color images and performs incremental mapping using COLMAP.
- Features are extracted from the images to create a COLMAP database.
- Feature matching is performed to establish correspondences between images.
- Incremental mapping is performed to estimate camera poses and generate a sparse 3D model.
- The camera poses are also extracted from the sparse model if needed. (Can be used with open3D to get specific viewpoints)

2. **Dense Reconstruction:**
- The dense reconstruction step takes the sparse reconstruction data and the color images to generate a dense point cloud. (Requires cuda compilation of COLMAP)
- COLMAP's patch match stereo algorithm is used to estimate depth maps for each image pair.
- The depth maps are then merged to create a dense point cloud representing the scene.
- This step provides a detailed representation of the scene geometry.

3. **Point Cloud Filtering:**
- The dense point cloud generated from the previous step may contain noise and outliers.
- Statistical outlier removal is applied to the point cloud to remove points that deviate significantly from their neighbors.
- This step helps to reduce noise and improve the quality of the point cloud.

4. **Point Cloud Segmentation:**
- The filtered point cloud is segmented into different objects using DBSCAN clustering.
- DBSCAN is a density-based clustering algorithm that groups together points that are closely packed.
- By adjusting the clustering parameters (`eps` and `min_points`), the script separates the point cloud into distinct objects.
- This step allows for individual processing and reconstruction of each object.

5. **Surface Reconstruction:**
- For each segmented object, surface reconstruction is performed using the Poisson surface reconstruction algorithm.
- Poisson surface reconstruction estimates the surface that best fits the point cloud by solving a Poisson equation.
- This step generates a triangular mesh representation of each object's surface.

6. **Mesh Refinement:**
- The reconstructed meshes may have some artifacts or noise.
- Laplacian and Taubin smoothing techniques are applied to refine the meshes.
- Laplacian smoothing smooths the mesh by averaging the positions of neighboring vertices.
- Taubin smoothing is a two-step smoothing process that helps to preserve mesh details while reducing noise.
- Additionally, degenerate triangles, duplicated vertices, and non-manifold edges are removed to improve mesh quality.

7. **Visualization and Merging:**
- The reconstructed objects are visualized using Open3D to provide a visual representation of the results.
- The individual object meshes are then merged into a single mesh using mesh concatenation.
- If needed, each object mesh can be individually saved as a `.ply` file to export to tools such as blender for further mesh refinement.
- The final merged mesh represents the complete reconstructed scene.

8. **Saving Results:**
- The final merged mesh is saved as `dense_reconstruction_mesh.ply` using the PLY file format.

## Example from RGB Scene
![One Frame from RGB video used to recreate scene]('Results/Sample Video Frame.png')
![Point Cloud Generated from Dense Reconstruction]('Results/Dense Point Cloud.png')
![Reconstruction with lower noise filtering to capture more of the scene]('Results/Higher Scene Capture.png')
![Complete Mesh with high noise filtering](Results/Mesh_Sample.png)
## Acknowledgements

This script utilizes the following libraries:

- [COLMAP](https://colmap.github.io/): A general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline.
- [pycolmap](https://github.com/colmap/pycolmap): Python bindings for COLMAP.
- [Open3D](http://www.open3d.org/): A modern library for 3D data processing.

The RGB-D images used to demonstrate the usage of this script are sourced from the following dataset:
- *Patch Volumes: Segmentation-based Consistent Mapping with RGB-D Cameras* by P. Henry, D. Fox, A. Bhowmik, R. Mongia, International Conference on 3D Vision (3DV), 2013.
