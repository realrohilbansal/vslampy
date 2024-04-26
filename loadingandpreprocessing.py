import sys
import cv2
import glob
import numpy as np
import os


def extractFramesFromVideo(video, output_dir="./img"):
    vid = cv2.VideoCapture(video)  # Captures the video
    frames = []  # list for storing captured frames
    downframes = []  # list for storing downsampled frames
    success = True
    count = 0  # frame count for storing purpose
    skip = 0  # skipping every alternative frame

    while (success):  # while the video has not ended/ is readable
        # Outputs a tuple of bool representing whether the video was read and a frame if it is true
        success, frame = vid.read()

        if success:
            if skip % 3 == 0:
                # converting the frame to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (640, 480),
                                   interpolation=cv2.INTER_AREA)
                frames.append(frame)  # store the grayframe in frames list
                # takes file and frame as input and encodes the frame in png
                cv2.imwrite(os.path.join(output_dir + "/frames",
                            f'frame{count}.png'), frame)

                downframe = cv2.pyrDown(frame)  # Gaussian blur then downsample
                downframes.append(downframe)
                cv2.imwrite(os.path.join(output_dir + "/downframes",
                            f'downframe{count}.png'), downframe)

                count += 1
            skip += 1

    vid.release()  # releasing the captured video for saving space

    return frames, downframes


# def readFrames(frames, intrinsic_camera_parameters_path):

# 	with open(intrinsic_camera_parameters_path) as f:
# 		content = f.readlines()
# 		# you may also want to remove whitespace characters like `\n` at the end of each line
# 		content = [x.strip() for x in content]

# 		first_line_split = content[0].split()
# 		second_line_split = content[1].split()

# 		# I believe these are right, given https://github.com/tum-vision/lsd_slam/blob/master/README.md#313-camera-calibration
# 		fx_width = float(first_line_split[0])
# 		fy_height = float(first_line_split[1])
# 		cx_width = float(first_line_split[2])
# 		cy_height = float(first_line_split[3])

# 		width = int(second_line_split[0])
# 		height = int(second_line_split[1])

# 	K = np.array([
# 		[np.round_(fx_width * width), 0, np.round_(cx_width * width)],
# 		[0, np.round_(fy_height * height), np.round_(cy_height * height)],
# 		[0, 0, 1]
# 	])

# 	# dimensions = (height, width)

# 	return images, K
if __name__ == '__main__':
    video = "video.mp4"

    frames, downframes = extractFramesFromVideo(video)
    print("Frames extracted from video")
    print(len(frames))
