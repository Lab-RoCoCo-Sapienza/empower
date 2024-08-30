#!/usr/bin/env python

import rospy
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from PIL import Image as PILImage

import cPickle as pickle  # For Python 2.x
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import cv2
import os
import time
import tf2_ros
import tf2_py as tf2
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from actionlib import SimpleActionClient

from paths import IMAGES_DIR, OUTPUT_DIR

rospy.init_node('image_processing', anonymous=True)
bridge = CvBridge()
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

USE_CASE = rospy.get_param('/use_case')
SPEECH = rospy.get_param('/speech')

SCAN_DIR = IMAGES_DIR+USE_CASE+'/'
DUMP_DIR = OUTPUT_DIR+USE_CASE+'/'

def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    height, width = depth_image.shape
    points = []

    v, u =  np.indices((height, width))

    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image

    points = np.dstack((x, y, z)).reshape(-1, 3)

    return points

def say_phrase(phrase):
    client = SimpleActionClient('/tts', TtsAction)
    client.wait_for_server()
    goal = TtsGoal()
    goal.rawtext.text = phrase
    goal.rawtext.lang_id = "en_GB"
    client.send_goal_and_wait(goal)


def listener():
    if SPEECH:
        say_phrase("Ok I will try to do it")
    msg_img = rospy.wait_for_message("/xtion/rgb/image_rect_color", Image)
    img = bridge.imgmsg_to_cv2(msg_img, "bgr8")
    img_path = SCAN_DIR+'scan.jpg'
    cv2.imwrite(img_path, img)

    msg_img_g = rospy.wait_for_message("/xtion/depth/image_raw", Image)
    camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
    proj_matrix = camera_info.K   

    fx = proj_matrix[0]
    fy = proj_matrix[4]
    cx = proj_matrix[2]
    cy = proj_matrix[5]

    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    img_g = bridge.imgmsg_to_cv2(msg_img_g)
    depth_image = np.asarray(img_g)

    point_cloud = depth_image_to_point_cloud(depth_image, camera_intrinsics)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    pcd.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))
  
    if not os.path.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)
    o3d.io.write_point_cloud(DUMP_DIR+"depth_pointcloud.pcd", pcd)
    
    # UNCOMMENT TO VISUALIZE PCD
    # o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    listener()

    