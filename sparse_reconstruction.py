import os
import numpy as np
import pycolmap
import shutil
import open3d as o3d
import cv2

def load_rgbd_images(color_file, depth_file):
    color_raw = o3d.io.read_image(color_file)
    depth_raw = o3d.io.read_image(depth_file)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
    
    return rgbd_image

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

# Load sparse reconstruction from COLMAP into Open3D point cloud
def get_point_cloud_from_sparse_model(sparse_model):
    points = []
    colors = []
    for point in sparse_model.points3D.values():
        points.append(point.xyz)
        colors.append(point.color)
    
    points = np.array(points)
    colors = np.array(colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to range [0, 1]
    
    return pcd

def filter_outliers(point_cloud, nb_neighbors=100, std_ratio=0.3):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = point_cloud.select_by_index(ind)
    return inlier_cloud

def segment_point_cloud(point_cloud, eps=0.05, min_points=20):
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    segments = []
    for i in range(max_label + 1):
        segment = point_cloud.select_by_index(np.where(labels == i)[0])
        segments.append(segment)
    return segments

def surface_reconstruction(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=25))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=10)
    return mesh

def refine_mesh(mesh):
    # Apply Laplacian smoothing
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)
    
    # Apply Taubin smoothing
    mesh = mesh.filter_smooth_taubin(number_of_iterations=1)
    
    # # Remove degenerate triangles
    # mesh.remove_degenerate_triangles()
    
    # # Remove duplicated vertices
    # mesh.remove_duplicated_vertices()
    
    # # Remove non-manifold edges
    # mesh.remove_non_manifold_edges()
    return mesh

def main():
    dataset_path = "rgbd-scenes-v2\imgs\scene_01"
    color_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith("-color.png")])
    depth_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith("-depth.png")])

    rgbd_images = []
    for color_file, depth_file in zip(color_files, depth_files):
        rgbd_image = load_rgbd_images(color_file, depth_file)
        rgbd_images.append(rgbd_image)
    
    print("Estimating Camera Pose")
    sparse_model, camera_poses = sparse_reconstruction(color_files)
    
    # print("Creating Point Cloud")
    print("Extracting Point Cloud from Sparse Model")
    point_cloud = get_point_cloud_from_sparse_model(sparse_model)

    print("Filtering Outliers")
    filtered_pcd = filter_outliers(point_cloud)
    o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

    print("Segmenting Point Cloud")
    segments = segment_point_cloud(filtered_pcd)

    print(f"Number of segments: {len(segments)}")

    meshes = []
    for i, segment in enumerate(segments):
        print(f"Reconstructing Segment {i+1}")
        mesh = surface_reconstruction(segment)
        filled_mesh = refine_mesh(mesh)
        meshes.append(filled_mesh)

    o3d.visualization.draw_geometries(meshes, window_name="Reconstructed Meshes")


if __name__ == "__main__":
    main()