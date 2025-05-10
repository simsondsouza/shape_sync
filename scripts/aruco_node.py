#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import tf.transformations as tf_transformations
from std_msgs.msg import Header
import tf.transformations as tft

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

class ArucoNode:
    def __init__(self):
        rospy.init_node('aruco_node')

        # Parameters
        self.marker_size = rospy.get_param("~marker_size", 0.055)
        dictionary_id_name = rospy.get_param("~aruco_dictionary_id", "DICT_5X5_250")
        image_topic = rospy.get_param("~image_topic", "/camera2/camera/color/image_raw")
        info_topic = rospy.get_param("~camera_info_topic", "/camera2/camera/color/camera_info")
        self.camera_frame = rospy.get_param("~camera_frame", "camera2_color_optical_frame")

        rospy.loginfo(f"Marker size: {self.marker_size}")
        rospy.loginfo(f"Marker type: {dictionary_id_name}")
        rospy.loginfo(f"Image topic: {image_topic}")
        rospy.loginfo(f"Camera info topic: {info_topic}")

        # Dictionary ID
        try:
            dictionary_id = getattr(aruco, dictionary_id_name)
        except AttributeError:
            rospy.logerr("Invalid aruco_dictionary_id: {}".format(dictionary_id_name))
            valid_options = [s for s in dir(aruco) if s.startswith("DICT")]
            rospy.logerr("Valid options: {}".format(", ".join(valid_options)))
            raise

        self.aruco_dictionary = aruco.getPredefinedDictionary(dictionary_id)
        # self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_parameters = aruco.DetectorParameters()
        self.bridge = CvBridge()

        # Subscribers
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None
        self.info_sub = rospy.Subscriber(info_topic, CameraInfo, self.info_callback)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

        # Publishers
        self.poses_pub = rospy.Publisher("aruco_poses", PoseArray, queue_size=10)
        self.image_with_markers_pub = rospy.Publisher("camera/image_with_markers", Image, queue_size=10)

    def info_callback(self, msg):
        self.info_msg = msg
        self.intrinsic_mat = np.reshape(np.array(msg.K), (3, 3))
        self.distortion = np.array(msg.D)
        # Unsubscribe after receiving camera info once
        self.info_sub.unregister()

    def image_callback(self, msg):
        if self.info_msg is None:
            rospy.logwarn("No camera info received yet.")
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detector = aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        corners, marker_ids, rejected = detector.detectMarkers(cv_image)

        pose_array = PoseArray()

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.camera_frame or self.info_msg.header.frame_id

        pose_array.header = header

        if marker_ids is not None:
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
                corners, self.marker_size, self.intrinsic_mat, self.distortion
            )

            for i, marker_id in enumerate(marker_ids):
                aruco.drawDetectedMarkers(cv_image, corners, marker_ids)
                cv_image =cv2.drawFrameAxes(cv_image, self.intrinsic_mat, self.distortion, rvecs[i], tvecs[i], 0.03, 3)
                # Pose calculations
                fixed_z = 0.080
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i]))[0]
                marker_position = np.array([tvecs[i][0], tvecs[i][1], fixed_z])
                rotated_position = np.dot(rotation_matrix[0:3, 0:3], marker_position)

                pose = Pose()
                pose.position.x = round(rotated_position[0][0], 2)
                pose.position.y = round(rotated_position[1][0], 2)
                pose.position.z = round(tvecs[0][2][0], 2)

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i]))[0]
                quat = tf_transformations.quaternion_from_matrix(rot_matrix)
                roll, pitch, yaw = tft.euler_from_quaternion(quat)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                pose_array.poses.append(pose)


            self.poses_pub.publish(pose_array)

        # Publish image with markers
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        img_msg.header = header
        self.image_with_markers_pub.publish(img_msg)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ArucoNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
