import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from autolab_core import RigidTransform, YamlConfig
from frankapy import FrankaArm
import time
import tf2_ros
import tf_conversions
import numpy as np
from geometry_msgs.msg import Point, Pose
import torch
import tf.transformations as tft
from geometry_msgs.msg import PoseArray

torch.cuda.empty_cache()
import threading

yaw = None
yaw_lock = threading.Lock()
yaw_received = threading.Event()
def aruco_pose_callback(msg):
    global yaw
    for pose in msg.poses:
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        roll, pitch, new_yaw = tft.euler_from_quaternion(quat)

        with yaw_lock:
            yaw = new_yaw
            yaw_received.set()  # Notify that yaw has been updated

        rospy.loginfo(f"Yaw received: {yaw}")

def load_tf_file(filename):
    with open(filename, 'r') as file:
        # Read the lines from the file
        lines = file.readlines()

    # Read frame names
    frames = lines[0].strip()
    tool = lines[1].strip()

    # Read transformation matrix (3x3 rotation matrix)
    matrix = np.array([list(map(float, line.split())) for line in lines[2:]])

    return frames, tool, matrix

# Example usage
filename1 = "/home/ros_ws/src/devel_packages/shape_sync/Shape-Detection/T_RSD415_franka.tf"  # replace with actual file path
filename2 = "/home/ros_ws/src/devel_packages/shape_sync/Shape-Detection/T_RS_mount_delta.tf"  # replace with actual file path

frames, tool, Tee = load_tf_file(filename1)
frames_, tool_, Tee_delta = load_tf_file(filename2)
Tee = RigidTransform(rotation=Tee[1:4, :3], translation=Tee[0,:3], from_frame='realsense', to_frame='franka_tool') 


# Load YOLO model
model_path = os.path.join(os.getcwd(), "object_detection_weights_50_epochs.pt")
model = YOLO(model_path)
model = model.cuda()

# Publisher for processed image
processed_image_pub = rospy.Publisher("/camera1/color/processed_image", Image, queue_size=1)

# Initialize the CvBridge
bridge = CvBridge()

# Camera intrinsic parameters (replace with actual calibration values)
fx = 926.4746704101562  # Focal length in x-axis
fy = 925.6631469726562  # Focal length in y-axis
cx = 627.109619140625  # Principal point in x-axis
cy = 369.2387390136719  # Principal point in y-axis

# Global variable for depth image
depth_image = None


def depth_image_callback(msg):
    """Callback to handle the depth image."""
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, "32FC1")  # Convert depth image to numpy array
    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

def get_depth_at_pixel(x, y):
    """Get depth value at pixel (x, y) in the depth image."""
    if depth_image is not None:
        return depth_image[y, x]  # Depth in meters
    else:
        rospy.logwarn("Depth image not yet available.")
        return None

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

def image_callback(msg):
    """ROS Image callback function."""
    try:
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        results = model(image)

        shapes = ['circle','square','triangle']
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                object_name = model.names[int(box.cls[0])]
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Only process if the detected object matches the user-specified shape
                for user_shape  in shapes:

                    rot = np.array([[ 1.0 ,-0.0 ,0.0],
                                                    [-0.0 ,-1.0  ,0.00],
                                                    [ 0.0, -0.00, -1.0]])
                    trans = np.array([0.51572232, 0.19906757, 0.43869236]) 
                    Move(rot, trans, fa)
                    if object_name.lower() == user_shape.lower():
                        # Crop detected object and detect knob (or relevant part of the object)
                        cropped_obj = image[y1:y2, x1:x2].copy()
                        knob_circle = detect_knob_center(cropped_obj)

                        # Draw bounding box and object center
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

                        # If knob center is detected, move the robot
                        if knob_circle is not None:
                            knob_x, knob_y, knob_radius = knob_circle
                            knob_x += x1  # Adjust to original image coordinates
                            knob_y += y1
                            cv2.circle(image, (knob_x, knob_y), knob_radius, (0, 0, 255), 2)
                            cv2.circle(image, (knob_x, knob_y), 5, (0, 0, 255), -1)
                            rospy.loginfo(f"Detected knob center at ({knob_x}, {knob_y}) in object {object_name}")
                            processed_msg = bridge.cv2_to_imgmsg(image, "bgr8")
                            processed_image_pub.publish(processed_msg)

                            z = 0.4386
                            x = -(knob_x - cx) * z / fx
                            y = -(knob_y - cy) * z / fy


                            T_ready_world = fa.get_pose()

                            transformed_coords= np.matmul(cam_to_world, np.array([z,x,y,1]).reshape(4,1))
                            x, y, z = transformed_coords[:3].flatten()
                            
                            rot_ = np.array([[ 0.9996638,   0.02375005,  0.00943208],
                            [ 0.02384119, -0.99965935, -0.00967084],
                            [ 0.00919918,  0.00989246, -0.99990875]])
                            
                            trans_ = np.array([x-0.015, y, 0.123 ])  # Replace with actual desired translation
                            Move(rot_, trans_, fa)
                            
                            rot = np.array([[ 0.9996638,   0.02375005,  0.00943208],
                            [ 0.02384119, -0.99965935, -0.00967084],
                            [ 0.00919918,  0.00989246, -0.99990875]])
                            trans = np.array([x-0.015, y, 0.021 ])  # Replace with actual desired translation
                            Move(rot, trans, fa)
                            fa.close_gripper()

                            fa.goto_joints([0,-0.7,0.0,-2.15,0,1.57,0.7])
                            if user_shape == 'circle':
                                rot1 = np.array([[ 1.0 ,-0.0 ,0.0],
                                        [-0.0 ,-1.0  ,0.00],
                                        [ 0.0, -0.00, -1.0]])
                                
                                trans1= np.array([ 0.37291951, -0.28049948,  0.29700775])

                                Move(rot1, trans1, fa)

                                
                                rot2 = np.array([[ 1.0 ,-0.0 ,0.0],
                                        [-0.0 ,-1.0  ,0.00],
                                        [ 0.0, -0.00, -1.0]])
                                trans2 = np.array([ 0.37291951, -0.28049948,  0.04760969])
                                Move(rot2, trans2, fa)
                                fa.open_gripper()

                                rot1 = np.array([[ 1.0 ,-0.0 ,0.0],
                                        [-0.0 ,-1.0  ,0.00],
                                        [ 0.0, -0.00, -1.0]])
                                
                                trans1= np.array([ 0.37291951, -0.28049948,  0.29700775])

                                Move(rot1, trans1, fa)
                                fa.reset_joints()

                            elif user_shape == 'square':
                                rot2 = np.array([[ 1.0 ,-0.0 ,0.0],
                                                    [-0.0 ,-1.0  ,0.00],
                                                    [ 0.0, -0.00, -1.0]])
                                trans2 = np.array([ 0.6184368,  -0.20051892 , 0.24975932])
                                Move(rot2, trans2, fa)     
                                if not yaw_received.wait(timeout=10.0):  # Wait up to 2 seconds
                                    rospy.logwarn("No ArUco yaw received in time, skipping yaw correction.")
                                    return
                                time.sleep(5)
                                with yaw_lock:
                                    current_yaw = yaw

                                # Proceed with yaw correction
                                current_joints = fa.get_joints()
                                current_joints[6] -= (1.57 - current_yaw)
                                rospy.loginfo(f"error: {1.57 - current_yaw}")
                                fa.goto_joints(current_joints)

                                rospy.sleep(3)
                                new_pose= fa.get_pose()
                                new_rot = new_pose.rotation

                                trans1= np.array([ 0.61828038 ,-0.28334818 , 0.29700775])

                                Move(new_rot, trans1, fa)

                                rot2 = np.array([[ 1.0 ,-0.0 ,0.0],
                                                    [-0.0 ,-1.0  ,0.00],
                                                    [ 0.0, -0.00, -1.0]])
                                trans2 = np.array([0.62828038, -0.28339948 , 0.04451141])
                                Move(new_rot, trans2, fa)
                                fa.open_gripper()

                                rot2 = np.array([[ 1.0 ,-0.0 ,0.0],
                                                    [-0.0 ,-1.0  ,0.00],
                                                    [ 0.0, -0.00, -1.0]])
                                trans2 = np.array([ 0.6184368,  -0.20051892 , 0.24975932])
                                Move(rot2, trans2, fa)  
                                fa.reset_joints()
                            

                            elif user_shape == 'triangle':
                                rot2 = np.array([[ 1.0 ,-0.0 ,0.0],
                                                    [-0.0 ,-1.0  ,0.00],
                                                    [ 0.0, -0.00, -1.0]])
                                trans2 = np.array([ 0.6184368,  -0.20051892 , 0.24975932])
                                Move(rot2, trans2, fa)     
                                print(fa.get_joints())
                                if not yaw_received.wait(timeout=10.0):  # Wait up to 2 seconds
                                    rospy.logwarn("No ArUco yaw received in time, skipping yaw correction.")
                                    return
                                time.sleep(5)
                                with yaw_lock:
                                    current_yaw = yaw

                                # Proceed with yaw correction
                                current_joints = fa.get_joints()
                                current_joints[6] -= (0.55 - current_yaw)
                                rospy.loginfo(f"error: {0.55 - current_yaw}")
                                fa.goto_joints(current_joints)

                                rospy.sleep(3)
                                new_pose= fa.get_pose()
                                new_rot = new_pose.rotation
                                
                                trans1= np.array([ 0.4871952 , -0.296183 , 0.29700775])
                                Move(new_rot, trans1, fa)

                                rot2 = np.array([[ 1.0 ,-0.0 ,0.0],
                                                    [-0.0 ,-1.0  ,0.00],
                                                    [ 0.0, -0.00, -1.0]])
                                trans2 = np.array([0.4871952 , -0.296183  ,0.04667181])
                                Move(new_rot, trans2, fa)
                                fa.open_gripper()

                                rot2 = np.array([[ 1.0 ,-0.0 ,0.0],
                                                    [-0.0 ,-1.0  ,0.00],
                                                    [ 0.0, -0.00, -1.0]])
                                trans2 = np.array([ 0.6184368,  -0.20051892 , 0.24975932])
                                Move(rot2, trans2, fa)     
                                fa.reset_joints()

        fa.reset_joints()
        rospy.signal_shutdown()
                            
        # Convert back to ROS image and publish
        print(image.shape)
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

# Main entry point
if __name__ == "__main__":
    # Initialize Franka arm
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    start = time.time()
    

    # Initialize ROS node
    rospy.Subscriber("/aruco_poses", PoseArray, aruco_pose_callback, queue_size=10)
    # Subscribe to depth image topic
    rospy.Subscriber("/camera1/camera/aligned_depth_to_color/image_raw", Image, depth_image_callback, queue_size=1)

    # Get user input for the shape to detect
    # user_shape = input("Enter the shape to detect (Diamond, Square, Rectangle, Triangle, Circle, Oval, Pentagon, Octagon): ")
    rot = np.array([[ 0.9996638,   0.02375005,  0.00943208],
                     [ 0.02384119, -0.99965935, -0.00967084],
                     [ 0.00919918,  0.00989246, -0.99990875]])
    trans = np.array([0.51572232, 0.19906757, 0.43869236]) 
    Move(rot, trans, fa)

    tfBuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfBuffer)
    trans = tfBuffer.lookup_transform("panda_link0", "camera1_color_frame", rospy.Time(), rospy.Duration.from_sec(0.5)).transform
    pose = Pose(position=Point(x=trans.translation.x, y=trans.translation.y, z=trans.translation.z), orientation=trans.rotation)
    cam_to_world = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

    # Start tracking objects after moving to initial position
    image_sub = rospy.Subscriber("/camera1/camera/color/image_raw", Image, lambda msg: image_callback(msg), queue_size=1, buff_size=2**24)

    # Start tracking objects
    rospy.spin()





