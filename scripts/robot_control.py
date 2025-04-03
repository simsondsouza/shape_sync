import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from autolab_core import RigidTransform
from frankapy import FrankaArm
import time
import pyrealsense2 as rs

# Load YOLO model
model_path = os.path.join(os.getcwd(), "object_detection_weights_50_epochs.pt")
model = YOLO(model_path)
model = model.cuda()

# Publisher for processed image
processed_image_pub = rospy.Publisher("/camera/color/processed_image", Image, queue_size=1)

# RealSense Camera setup (assuming depth and color are aligned)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Camera intrinsic parameters (replace with actual calibration values)
fx = 926.4746704101562  # Focal length in x-axis
fy = 925.6631469726562  # Focal length in y-axis
cx = 627.109619140625 # Principal point in x-axis
cy = 369.2387390136719  # Principal point in y-axis

def get_depth_at_pixel(x, y):
    """Get depth value at pixel (x, y) in the image."""
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth = depth_frame.get_distance(x, y)  # Depth in meters
    return depth

def detect_knob_center(cropped_img):
    """Detects the center of a knob in the cropped object region."""
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([5, 80, 10])
    upper_brown = np.array([15, 255, 190])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    blurred = cv2.GaussianBlur(mask, (5, 5), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=60, param2=30, minRadius=5, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]  # Return the first detected circle (x, y, radius)
    return None

def Move(rot, trans, fa):
    """Moves the robot end-effector to the given position and orientation."""
    des_pose = RigidTransform(rotation=rot, translation=trans, from_frame='franka_tool', to_frame='world') 
    fa.goto_pose(des_pose, use_impedance=True)

def image_callback(msg, user_shape):
    """ROS Image callback function."""
    try:
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                object_name = model.names[int(box.cls[0])]
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Only process if the detected object matches the user-specified shape
                if object_name.lower() == user_shape.lower():
                    # Crop detected object and detect knob (or relevant part of the object)
                    cropped_obj = image[y1:y2, x1:x2].copy()
                    knob_circle = detect_knob_center(cropped_obj)

                    # Draw bounding box and object center
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
                    cv2.putText(image, f"{object_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # If knob center is detected, move the robot
                    if knob_circle is not None:
                        knob_x, knob_y, knob_radius = knob_circle
                        knob_x += x1  # Adjust to original image coordinates
                        knob_y += y1
                        cv2.circle(image, (knob_x, knob_y), knob_radius, (0, 0, 255), 2)
                        cv2.circle(image, (knob_x, knob_y), 5, (0, 0, 255), -1)
                        rospy.loginfo(f"Detected knob center at ({knob_x}, {knob_y}) in object {object_name}")

                        # Get depth at knob center
                        depth = get_depth_at_pixel(knob_x, knob_y)

                        # Convert image coordinates to camera frame (3D coordinates)
                        z = depth
                        x = (knob_x - cx) * z / fx
                        y = (knob_y - cy) * z / fy

                        rospy.loginfo(f"Knob in camera frame: ({x}, {y}, {z})")

                        # Now transform from camera frame to robot's end-effector frame
                        # Assuming we have a static transformation matrix from camera to robot
                        camera_to_robot_tf = np.array([
                            [0.999, 0.0, 0.0, 0.5],  # Replace with actual transformation
                            [0.0, 0.999, 0.0, 0.0],  # Replace with actual transformation
                            [0.0, 0.0, 1.0, 0.2],    # Replace with actual transformation
                            [0.0, 0.0, 0.0, 1.0]
                        ])
                        object_in_camera_frame = np.array([x, y, z, 1.0])
                        object_in_robot_frame = np.dot(camera_to_robot_tf, object_in_camera_frame)

                        rospy.loginfo(f"Object in robot frame: {object_in_robot_frame[:3]}")

                        # Use inverse kinematics to move the robot's end-effector to this position
                        rot = np.eye(3)  # Assuming no rotation (or replace with actual desired rotation)
                        trans = object_in_robot_frame[:3]
                        Move(rot, trans, fa)

        # Convert back to ROS image and publish
        # processed_msg = bridge.cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), "bgr8")
        processed_msg = bridge.cv2_to_imgmsg(image, "bgr8")
        processed_image_pub.publish(processed_msg)
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

# Main entry point
if __name__ == "__main__":
    # Initialize Franka arm
    fa = FrankaArm()
    fa.reset_joints()
    start = time.time()

    # Initialize ROS node
    # rospy.init_node('yolo_object_detector')
    bridge = CvBridge()
    
    # Get user input for the shape to detect
    user_shape = input("Enter the shape to detect (Diamond, Square, Rectangle, Triangle, Circle, Oval, Pentagon, Octagon): ")

    # Move robot to initial position
    rot = np.array([[ 0.9996638,   0.02375005,  0.00943208],
                     [ 0.02384119, -0.99965935, -0.00967084],
                     [ 0.00919918,  0.00989246, -0.99990875]])
    trans = np.array([0.51572232, 0.19906757, 0.43869236])
    Move(rot, trans, fa)

    # Start tracking objects after moving to initial position
    rospy.sleep(1)  # Give time for robot to reach position

    # Subscribe to RealSense camera image topic
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, lambda msg: image_callback(msg, user_shape), queue_size=1, buff_size=2**24)

    # Keep the program running
    rospy.spin()
