import cv2
import numpy as np

print("Sahil ghusa")
def semi_dense_depth_estimation(image, reference_image, window_size, min_disp=0, max_disp=128):
    """
    Estimates a semi-dense depth map for the given image using the reference image.

    Args:
        image: The current image for which to estimate depth (grayscale).
        reference_image: The reference image used for depth propagation (grayscale).
        window_size: Half-size of the search window for cost volume calculation.
        min_disp: Minimum possible disparity value (default: 0).
        max_disp: Maximum possible disparity value (default: 128).

    Returns:
        depth_map: The estimated depth map for the image (in meters, assuming calibrated camera).
    """

    # Preprocessing
    image = image.astype(np.float32) / 255.0  # Normalize image intensities
    reference_image = reference_image.astype(np.float32) / 255.0

    # Cost Volume Calculation
    cost_volume = np.zeros(
        (image.shape[0], image.shape[1], max_disp - min_disp + 1))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for d in range(min_disp, max_disp + 1):
                ref_x = x - d
                if ref_x < 0 or ref_x >= reference_image.shape[1]:
                    cost = np.inf  # Penalize out-of-image access
                else:
                    cost = (image[y, x] - reference_image[y, ref_x])**2
                cost_volume[y, x, d - min_disp] = cost

    # Semi-Global Matching (SGM)
    disparity_map = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        # Start from second column (avoid image border)
        for x in range(1, image.shape[1]):
            # Left-to-right pass (accumulation)
            min_cost = cost_volume[y, x, 0]
            min_disp = 0
            for d in range(1, max_disp - min_disp + 1):
                cost = cost_volume[y, x, d] + min_cost - \
                    cost_volume[y, x - 1, d - 1]
                if cost < min_cost:
                    min_cost = cost
                    min_disp = d
            disparity_map[y, x] = min_disp

            # Right-to-left pass (correction)
            min_cost = cost_volume[y, x, max_disp - min_disp]
            for d in range(max_disp - min_disp - 1, -1, -1):
                cost = cost_volume[y, x, d] + min_cost - \
                    cost_volume[y, x + 1, d + 1]
                if cost < min_cost:
                    min_cost = cost
                    min_disp = d
            # Average accumulation & correction
            disparity_map[y, x] = (disparity_map[y, x] + min_disp) / 2

    # Disparity Refinement (Bilateral Filtering)
    disparity_map = cv2.bilateralFilter(disparity_map, 9, 75, 75)

    # Depth Map Generation (assuming calibrated camera)
    focal_length =  # Replace with your camera's focal length (in meters)
    baseline =  # Replace with your camera's baseline (in meters)
    depth_map = focal_length * baseline / \
        (disparity_map + min_disp)  # Invert disparity for depth

    return depth_map


# Example usage (assuming you have loaded your images)
depth_map = semi_dense_depth_estimation(image, reference_image, window_size=5)

# Visualize or further process the depth map
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()


def calculate_photometric_error(image1, image2):
    """Calculates the photometric error between two images (MSD)."""
    diff = image1.astype(np.float32) - image2.astype(np.float32)
    return np.mean(diff**2)


def optimize_pose(reference_image, current_image):
    """Optimizes the camera pose using PnP with RANSAC (replace with your preferred method)."""
    # Assuming grayscale images and known camera intrinsics (focal length, etc.)
    sift = cv2.xfeatures2d.SIFT_create()
    # Extract and match keypoints
    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kp2, des2 = sift.detectAndCompute(current_image, None)
    matches = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    good_matches = matches.knnMatch(des1, des2, k=2)
    # Filter good matches based on Lowe's ratio test (optional)
    good_matches = [m for m, n in good_matches if m.distance < 0.7*n.distance]
    if len(good_matches) > 4:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # Estimate pose using RANSAC
        ret, rotation_vec, translation_vec, mask = cv2.solvePnPRansac(
            src_pts, dst_pts, camera_matrix, distortion_coefficients, None, None, reprojectionError=5, confidence=0.99, mask=None)
        if ret:
            return rotation_vec, translation_vec
    return None, None  # Return None for failed pose estimation


def accumulate_error(reference_image, current_image, pose, error_map=None):
    """Accumulates photometric error during pose refinement."""
    # ... (code from previous section)

    return error_map


def filter_depth_map(error_map, threshold):
    """Filters the accumulated error to create a semi-dense depth map."""
    filtered_depth_map = error_map < threshold
    return filtered_depth_map


def select_keyframe(reference_image, current_image, last_error, min_threshold=0.1, max_interval=30):
    """Selects keyframes based on photometric error change and time interval."""
    error = calculate_photometric_error(reference_image, current_image)
    is_keyframe = error > last_error + \
        min_threshold or (not last_keyframe and len(keyframes) >= max_interval)
    return is_keyframe, error


def main():
    """Main loop for processing an image sequence."""
    # Load images and camera intrinsics (focal length, distortion coefficients)
    images = []
    for filename in image_filenames:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    camera_matrix = np.array(
        [[focal_length, 0, image_width // 2], [0, focal_length, image_height // 2], [0, 0, 1]])
    # Assuming no distortion for simplicity
    distortion_coefficients = np.zeros((4, 1))

    # Initialize variables
    reference_image = images[0]
    last_error = np.inf
    last_keyframe = False
    keyframes = []

    for current_image in images[1:]:
        # Keyframe selection
        is_keyframe, current_error = select_keyframe(
            reference_image, current_image, last_error)

        if is_keyframe:
            # Optimize pose
            rotation_vec, translation_vec = optimize_pose(
                reference_image, current_image)
            # if rotation_vec is not None and translation_vec is not None:
            # Accumulate error


'''Pose graph creation and depth fusion'''
# Initialize pose graph and depth map
pose_graph = PoseGraph()
global_depth_map = None

# Process each frame
for i, image in enumerate(images):
    # Estimate depth map and pose
    depth_map = estimate_depth_map(image)
    pose = estimate_pose(image)

    # Add pose to pose graph
    pose_graph.add_node(i, pose)
    if i > 0:
        pose_graph.add_edge(i-1, i, relative_pose)

    # Detect loop closure (optional)
    if detect_loop_closure(image):
        loop_closure_frame = find_loop_closure_frame(image)
        pose_graph.add_edge(i, loop_closure_frame, relative_pose)

    # Optimize pose graph
    optimized_pose = pose_graph.optimize()

    # Fuse depth map
    global_depth_map = fuse_depth_map(
        global_depth_map, depth_map, optimized_pose)

# Visualize depth map
visualize_depth_map(global_depth_map)


def transform_depth_map(depth_map, pose):
    """Transforms the depth map to the global coordinate system using the pose."""
    # This is a simplified version and assumes that the depth map is a 2D array
    # and the pose is a 4x4 transformation matrix. In practice, you would need
    # to handle the camera intrinsics and extrinsics, and the depth map might
    # be a 3D array or a point cloud.
    transformed_depth_map = np.dot(pose, depth_map)
    return transformed_depth_map


def fuse_depth_map(global_depth_map, depth_map, pose):
    """Fuses the depth map with the global depth map."""
    # Transform the depth map to the global coordinate system
    transformed_depth_map = transform_depth_map(depth_map, pose)

    # If the global depth map is None, initialize it with the transformed depth map
    if global_depth_map is None:
        global_depth_map = transformed_depth_map
    else:
        # Fuse the transformed depth map with the global depth map
        # This is a simple averaging method, but in practice you might want to
        # use a more sophisticated method like TSDF fusion
        global_depth_map = (global_depth_map + transformed_depth_map) / 2

    return global_depth_map
