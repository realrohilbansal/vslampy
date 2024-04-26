import cv2
import numpy as np
import g2o


def warp_image(image, pose):
    """
    Warps an image based on the estimated camera pose (translation and rotation).

    Args:
        image: The input image as a NumPy array.
        pose: A tuple representing the camera pose (tx, ty, rot_x, rot_y, rot_z).
            - tx, ty: Translation in x and y directions (pixels).
            - rot_x, rot_y, rot_z: Rotations around x, y, and z axes (radians).

    Returns:
        The warped image as a NumPy array.
    """

    # Extract pose parameters
    tx, ty, rot_x, rot_y, rot_z = pose

    # Get image dimensions
    rows, cols = image.shape[:2]

    # Create a transformation matrix based on pose
    # This example uses a full 3D transformation with rotation around all axes
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    rotation_matrix = cv2.Rodrigues(np.array([rot_x, rot_y, rot_z]))[0]
    transformation_matrix = np.matmul(rotation_matrix, translation_matrix)

    # Apply perspective transformation
    #  useful for larger camera motions or to account for depth information
    #  might need to estimate focal length for this step
    # focal_length = ...  # (Replace with your estimated focal length)
    # camera_center = (cols // 2, rows // 2)  # Assuming center is image center
    # perspective_matrix = np.float32([[focal_length, 0, camera_center[0]],
    #                                [0, focal_length, camera_center[1]],
    #                                [0, 0, 1]])
    # transformation_matrix = np.matmul(perspective_matrix, transformation_matrix)

    # Apply transformation to warp the image
    # Use cv2.warpPerspective for full 3D transformation (if perspective_matrix is used)
    # Otherwise, use cv2.warpAffine for simpler 2D affine transformation
    # Choose the appropriate function based on your needs
    warped_image = cv2.warpPerspective(
        image, transformation_matrix, (cols, rows))
    # warped_image = cv2.warpAffine(image, transformation_matrix, (cols, rows))

    return warped_image


def photometric_error(image1, image2, error_type="ssd"):
    """
    Calculates the photometric error between two images.

    Args:
        image1: The first image as a NumPy array.
        image2: The second image as a NumPy array.
        error_type: The type of error metric to use (default: "ssd").
            - "ssd": Sum of Squared Differences
            - "sad": Sum of Absolute Differences
            - "ncc": Normalized Cross Correlation (normalized to [-1, 1])

    Returns:
        The calculated photometric error.
    """

    # Check image dimensions for compatibility
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Cast images to floating point for calculations (if necessary)
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Calculate error based on the chosen metric
    if error_type == "ssd":
        error = np.sum(np.square(image1 - image2))
    elif error_type == "sad":
        error = np.sum(np.abs(image1 - image2))
    elif error_type == "ncc":
        # Normalize images (assuming zero mean and unit standard deviation)
        image1_normalized = (image1 - np.mean(image1)) / np.std(image1)
        image2_normalized = (image2 - np.mean(image2)) / np.std(image2)
        # Calculate NCC (normalized to range [-1, 1])
        error = np.mean(image1_normalized * image2_normalized)
    else:
        raise ValueError(
            "Invalid error_type. Choose from 'ssd', 'sad', or 'ncc'.")

    return error


def estimate_pose(reference_image, current_image, max_iterations=10, optimization_method="lm"):
    """
    Estimates the camera pose by minimizing the photometric error between the warped reference image and the current image.

    Args:
        reference_image: The reference image as a NumPy array.
        current_image: The current image as a NumPy array.
        max_iterations: Maximum number of iterations for optimization (default: 10).
        optimization_method: Optimization method to use (default: "lm" - Levenberg-Marquardt).

    Returns:
        The estimated camera pose as a tuple (tx, ty, rot_x, rot_y, rot_z).
    """

    # Initial guess for pose (adjust based on your expectations)
    initial_pose = (0, 0, 0, 0, 0)  # Minimal translation and rotation

    # Define optimization function based on chosen method
    if optimization_method == "lm":
        def optimize(pose):
            warped_image = warp_image(reference_image, pose)
            return photometric_error(warped_image, current_image)
    else:
        raise ValueError(
            "Invalid optimization_method. Choose 'lm' for Levenberg-Marquardt.")

    # Perform optimization to minimize photometric error
    best_pose, _, _ = cv2.solvePnP(None, None, None, None, initial_pose, max_iterations,
                                   criteria=cv2.TermCriteria(
                                       cv2.TERM_EPS | cv2.TERM_ITER, 1e-5, 1e-4),
                                   flags=cv2.SOLVEPNP_LEVENBERG_MARQUARDT if optimization_method == "lm" else 0)

    return best_pose


# Example usage
reference_image = cv2.imread("reference.png")
current_image = cv2.imread("current.png")

estimated_pose = estimate_pose(reference_image, current_image)
print("Estimated pose:", estimated_pose)
