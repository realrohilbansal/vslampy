import numpy as np
import cv2 
import scipy.optimize
from pose2 import warp_image, photometric_error

class Keyframe:
    """Represents a keyframe with pose and (optional) depth information."""

    def __init__(self, image, pose, ds_image = None, depth_map=None):
        self.k_id = 0 
        self.image = image # frame
        self.ds_image = ds_image
        self.pose = pose  # Pose as a 4x4 transformation matrix, estimated using warping
        self.depth_map = depth_map  # Optional depth map
        self.keyframe = False

class Edge:
    """Represents an edge connecting two keyframes with relative transformation."""

    def __init__(self, from_keyframe, to_keyframe, relative_transform = None):
        self.from_keyframe = from_keyframe
        self.to_keyframe = to_keyframe
        # Relative transformation as a 4x4 matrix
        self.relative_transform = relative_transform

class PoseGraph:
    """Represents a pose graph with vertices (keyframes) and edges."""

    def __init__(self):
        self.keyframes = []
        self.edges = []
        self.global_error = 0.0
        # Placeholder for accumulated error map (to be populated)
        self.error_map = None

    def add_keyframe(self, keyframe):
        """Adds a keyframe to the pose graph."""
        self.keyframes.append(keyframe)

    def connect_keyframes(self, from_keyframe, to_keyframe, relative_transform):
        """Connects two keyframes with an edge representing their relative transformation."""
        # from_keyframe = self.keyframes(from_keyframe)
        # to_keyframe = self.keyframes(to_keyframe)
        self.edges.append(
            Edge(from_keyframe, to_keyframe, relative_transform))
    
    def accumulate_error(self, constraint = 1.0):
        global_error = 0.0
        for edge in self.edges:
            from_keyframe = edge.from_keyframe
            to_keyframe = edge.to_keyframe
            warped_image = warp_image(from_keyframe.image, from_keyframe.pose, i=2)
            error = photometric_error(warped_image, to_keyframe.image)
            global_error += error
        # constraint = self.poseConstraint()
        global_error = global_error + constraint
        self.global_error = global_error
        return global_error
    
    # def poseConstraint(self):
    #     pass


    # def optimization(self):
    #     pose = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # Minimal translation and rotation
    #     pass

    def estimate_pose(self, i, accumulate_error):
        """
        Estimates the optimal pose based on photometric error.

        Args:
            reference_image: The original reference image (numpy array).
            initial_pose: The initial guess for the pose (6-element vector: [tx, ty, tz, rx, ry, rz]).
            warp_function: A function that takes the pose as input and returns the warped image.
            photometric_error_function: A function that takes the original and warped images as input 
                                        and returns the photometric error.

        Returns:
            optimized_pose: The optimized pose vector (6 elements) after pose optimization.
        """

        def objective_function():
            """
            Objective function for optimization (minimizes global photometric error).
            """
            # error = photometric_error(reference_image, warped_image)
            # errors = sliding_window_errors(reference_image, 32, 16, warped_image, i)
            global_error = accumulate_error()
            # print(errors[:5])
            # print(np.shape(errors))
            return global_error

        # Perform Nelder Mead optimization using minimize from SciPy
        initial 
        result = scipy.optimize.minimize(objective_function,  method="Nelder-Mead")

        # Extract the optimized pose from the result
        optimized_pose = result

        return optimized_pose
        

posegraph = PoseGraph()

image1 = cv2.imread("img/frames/frame500.png")
image2 = cv2.imread("img/frames/frame501.png")
pose1 = (5.0, 6.0, 10.9, 5.5, 7.9, 0.9)
pose2 = (10.0, 1.5, 7.7, 8.2, 6.0, 3.5 )
keyframe1 = Keyframe(image1, pose1)
keyframe2 = Keyframe(image2, pose2)

posegraph = PoseGraph()
posegraph.add_keyframe(keyframe1)
posegraph.add_keyframe(keyframe2)
posegraph.connect_keyframes(keyframe1, keyframe2, keyframe1.pose)

optimized_pose = posegraph.estimate_pose(
    posegraph.edges, 1, posegraph.accumulate_error)
print(optimized_pose)
