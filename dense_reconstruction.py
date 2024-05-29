import os
import numpy as np
import pycolmap
import open3d as o3d
import shutil

def sparse_reconstruction(color_files):
    # Set up paths
    database_path = "database.db"
    image_dir = "images"
    output_path = "sparse"

    # Create the image directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    # Copy color images to the image directory
    for i, color_file in enumerate(color_files):
        image_name = f"image{i+1}.png"
        image_path = os.path.join(image_dir, image_name)
        shutil.copyfile(color_file, image_path)

    # Extract features to create COLMAP Database
    print("Extracting Features")
    if not os.path.exists(database_path):
        # Extract features
        pycolmap.extract_features(database_path, image_dir)
    else:
        print("Database file already exists. Skipping feature extraction.")

    print("Matching Features")
    # Perform feature matching
    pycolmap.match_exhaustive(database_path)

    # Create a sparse reconstruction
    print("Performing Incremental Mapping")
    reconstruction = pycolmap.Reconstruction()
    pycolmap.incremental_mapping(database_path, image_dir, output_path)

    # Load the reconstructed sparse model
    sparse_model = pycolmap.Reconstruction("sparse/0") # May be necessary to change number depending on what run of sparse reconstruction you are on
    print(sparse_model.summary())

    # Extract camera poses from the sparse model
    camera_poses = []
    camera_poses = []
    for image_id, image in sparse_model.images.items():
        cam_from_world = image.cam_from_world
        # Extract rotation matrix and translation vector
        R = cam_from_world.rotation
        t = cam_from_world.translation
        # Create the camera pose matrix
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R.matrix()
        camera_pose[:3, 3] = t
        camera_poses.append(camera_pose)

    return sparse_model, camera_poses

def run_dense_reconstruction(sparse_dir, image_dir, dense_dir):
    
    # Check if the dense reconstruction has already been performed
    dense_point_cloud_path = os.path.join(dense_dir, "dense.ply")
    if os.path.exists(dense_point_cloud_path):
        # If the dense point cloud exists, load it directly
        pcd = o3d.io.read_point_cloud(dense_point_cloud_path)
        return pcd
    
    # If the dense reconstruction has not been performed, proceed with the reconstruction
    if not os.path.exists(dense_dir):
        os.makedirs(dense_dir)
    
    # Image undistortion
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=sparse_dir,
        image_path=image_dir,
        output_type='COLMAP'
    )
    
    # Patch match stereo
    pycolmap.patch_match_stereo(
        workspace_path=dense_dir,
        workspace_format='COLMAP',
        pmvs_option_name='option-all',
        options=pycolmap.PatchMatchOptions(),
        config_path=''
    )
    
    pycolmap.stereo_fusion(os.path.join(dense_dir, "dense.ply"), dense_dir)

    # Load the dense point cloud
    pcd = o3d.io.read_point_cloud(dense_point_cloud_path)
    
    return pcd

# Filter point cloud to remove noise and segment into different objects
def filter_and_segment_point_cloud(pcd):

    # Remove statistical outliers
    # If you want to capture more of the scene, increase the standard deviation ratio
    # Note: This will also increase the noise in your resulting meshes
    filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)

    # downsampled_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.005)  # Adjust the voxel size as needed

    o3d.visualization.draw_geometries([filtered_pcd], window_name=f"Filtered PCD")

    # Segment the objects using DBSCAN clustering
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(filtered_pcd.cluster_dbscan(eps=0.05, min_points=500, print_progress=True))

    # Get the unique object labels (excluding the background label)
    object_labels = np.unique(labels)
    object_labels = object_labels[object_labels != -1]

    # Extract each object cluster and store them in a list
    object_pcds = []
    for label in object_labels:
        object_pcd = filtered_pcd.select_by_index(np.where(labels == label)[0])
        object_pcds.append(object_pcd)

    return object_pcds

# Create a mesh of surface from point cloud
def surface_reconstruction(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=100))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=12)
    return mesh

def refine_mesh(mesh):
    # Smooth Mesh
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    # Remove duplicated vertices
    mesh.remove_duplicated_vertices()
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    return mesh

# Combine all the object meshes into singular mesh
def merge_meshes(meshes):
    merged_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        merged_mesh += mesh
    return merged_mesh

def main():
    image_dir = "images"
    sparse_dir = "sparse/0" # Update based on scene
    dense_dir = "dense"

    dataset_path = "rgbd-scenes-v2\imgs\scene_01"
    color_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith("-color.png")])
    depth_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith("-depth.png")])

    # Perform sparse reconstruction only if sparse representation does not already exist
    if not os.path.exists(sparse_dir):
        sparse_model, _ = sparse_reconstruction(color_files)
    else:
        print("Sparse reconstruction already exists. Skipping sparse reconstruction.")

    print("Running Dense Reconstruction")
    dense_pcd = run_dense_reconstruction(sparse_dir, image_dir, dense_dir)

    print("Filtering and Segmenting Point Cloud")
    object_pcds = filter_and_segment_point_cloud(dense_pcd)

    print("Performing Surface Reconstruction and Refinement")
    meshes = []
    for i, object_pcd in enumerate(object_pcds):
        print(f"Reconstructing Object {i+1}")
        mesh = surface_reconstruction(object_pcd)
        refined_mesh = refine_mesh(mesh)
        meshes.append(refined_mesh)

    print("Visualizing and Saving Meshes")
    o3d.visualization.draw_geometries(meshes, window_name="Reconstructed Objects")

    final_mesh = merge_meshes(meshes)
    o3d.io.write_triangle_mesh("dense_reconstruction_mesh.ply", final_mesh)

if __name__ == "__main__":
    main()
